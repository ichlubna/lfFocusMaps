#include <stdexcept>
#define GLM_FORCE_SWIZZLE
#include <sstream>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>
#include "interpolator.h"
#include "kernels.cu"
#include "lfLoader.h"
#include "libs/loadingBar/loadingbar.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "libs/stb_image_write.h"

class Timer
{
    public:
    Timer()
    {    
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent);
    }
    float stop()
    {
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
        float time = 0;
        cudaEventElapsedTime(&time, startEvent, stopEvent);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return time; 
    };
    private:
    cudaEvent_t startEvent, stopEvent;
};

Interpolator::Interpolator(std::string inputPath, std::string mode, bool useSecondary, bool mips, bool yuv, float spacingAspect) : useSecondaryFolder{useSecondary}, useMips{mips}, useYUV{yuv}, inputCamerasSpacingAspect{spacingAspect}, input{inputPath}
{
    addressMode = parseAddressMode(mode);
    init();
}

Interpolator::~Interpolator()
{
    cudaDeviceReset();
}

void Interpolator::init()
{
    loadGPUData();
    sharedSize = 0;//sizeof(half)*colsRows.x*colsRows.y;
}

int Interpolator::createTextureObject(const uint8_t *data, glm::ivec3 size)
{
    cudaTextureAddressMode cudaAddressMode = cudaAddressModeClamp;
    switch (addressMode)
    {
    case WRAP:
        cudaAddressMode = cudaAddressModeWrap;
    break;
    case CLAMP:
        cudaAddressMode = cudaAddressModeClamp;
    break;
    case MIRROR:
        cudaAddressMode = cudaAddressModeMirror;
    break;
    case BORDER:
        cudaAddressMode = cudaAddressModeBorder;
    break;
    case BLEND:
    case ALTER:
        cudaAddressMode = cudaAddressModeClamp;
    break;
    }

    size_t pitch = size.x*size.z;
    void *imageData;
    cudaMalloc(&imageData, size.x*size.y*size.z);
    cudaMemcpy2D(imageData, pitch, data, pitch, pitch, size.y, cudaMemcpyHostToDevice);

    if(useYUV)
        runKernel(ARR_RGB_YUV, {imageData, size.x, size.y});

    cudaChannelFormatDesc channels = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray *arr;
    cudaMallocArray(&arr, &channels, size.x, size.y);
    cudaMemcpy2DToArray(arr, 0, 0, imageData, pitch, pitch, size.y, cudaMemcpyDeviceToDevice);   
    cudaFree(imageData);
 
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = arr;
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.normalizedCoords = true;
    texDescr.addressMode[0] = cudaAddressMode;
    texDescr.addressMode[1] = cudaAddressMode;
    texDescr.readMode = cudaReadModeNormalizedFloat;
    cudaTextureObject_t texObj{0};
    cudaCreateTextureObject(&texObj, &texRes, &texDescr, NULL);
    return texObj;
}

std::pair<int, int*> Interpolator::createSurfaceObject(glm::ivec3 size, const uint8_t *data, bool copyFromDevice)
{
    auto arr = loadImageToArray(data, size, copyFromDevice);
    cudaResourceDesc surfRes;
    memset(&surfRes, 0, sizeof(cudaResourceDesc));
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = reinterpret_cast<cudaArray*>(arr);
    cudaSurfaceObject_t surfObj = 0;
    cudaCreateSurfaceObject(&surfObj, &surfRes);
    return {surfObj, arr};
}

int* Interpolator::loadImageToArray(const uint8_t *data, glm::ivec3 size, bool copyFromDevice)
{
    cudaChannelFormatDesc channels = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned); 
    if(size.z == 1)
        channels = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned); 
    cudaArray *arr;
    cudaMallocArray(&arr, &channels, size.x*sizeof(float), size.y, cudaArraySurfaceLoadStore);
    if(data != nullptr)
    {
        if(copyFromDevice)
            cudaMemcpy2DArrayToArray(arr, 0, 0, reinterpret_cast<cudaArray_const_t>(data), 0, 0, size.x*sizeof(float), size.y, cudaMemcpyDeviceToDevice); 
        else
            cudaMemcpy2DToArray(arr, 0, 0, data, size.x*size.z, size.x*size.z, size.y, cudaMemcpyHostToDevice);
    }
    return reinterpret_cast<int*>(arr);
}

Interpolator::TexturesInfo Interpolator::loadTextures(std::string input, void **textures)
{
    LfLoader lfLoader;
    lfLoader.loadData(input);
    auto textColsRows = lfLoader.getColsRows();
    auto textResolution = lfLoader.imageResolution();

    std::cout << "Uploading content of "+input+" to GPU..." << std::endl;
    LoadingBar bar(lfLoader.imageCount());
    
    std::vector<cudaTextureObject_t> textureObjects;
    for(int col=0; col<textColsRows.x; col++)
        for(int row=0; row<textColsRows.y; row++)
        { 
            textureObjects.push_back(createTextureObject(lfLoader.image({col, row}).data(), textResolution)); 
            bar.add();
        }

    cudaMalloc(textures, textureObjects.size()*sizeof(cudaTextureObject_t));
    cudaMemcpy(*textures, textureObjects.data(), textureObjects.size()*sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice); 
    return {textColsRows, textResolution};
}

void Interpolator::loadGPUData()
{
    std::string path = input;
    if(path.back() == '/')
        path.pop_back();

    auto info = loadTextures(input, &textureObjectsArr);
    colsRows = info.colsRows;
    resolution = info.resolution;

    if(useSecondaryFolder)
        loadTextures(path+"_sec", &secondaryTextureObjectsArr);
    else
        secondaryTextureObjectsArr = textureObjectsArr;
    
    if(useMips)
        loadTextures(path+"_down", &mipTextureObjectsArr);
    else
        mipTextureObjectsArr = textureObjectsArr;
 
    std::vector<cudaSurfaceObject_t> surfaces;
    for(int i=0; i<FileNames::OUTPUT_COUNT; i++)
    {
        auto surfaceRes = resolution;
        if(i == FileNames::FOCUS_MAP || i == FileNames::FOCUS_MAP_POST)
            surfaceRes.z = 1;
        auto surface = createSurfaceObject(surfaceRes);
        surfaces.push_back(surface.first);  
        surfaceOutputArrays.push_back(surface.second);
    }
    cudaMalloc(&surfaceObjectsArr, surfaces.size()*sizeof(cudaTextureObject_t));
    cudaMemcpy(surfaceObjectsArr, surfaces.data(), surfaces.size()*sizeof(cudaSurfaceObject_t), cudaMemcpyHostToDevice);
}

void Interpolator::loadGPUConstants(InterpolationParams params)
{
    std::vector<int> intValues(IntConstantIDs::INT_CONSTANTS_COUNT);
    intValues[IntConstantIDs::IMG_RES_X] = resolution.x;
    intValues[IntConstantIDs::IMG_RES_Y] = resolution.y;
    intValues[IntConstantIDs::GRID_SIZE] = colsRows.x*colsRows.y;
    intValues[IntConstantIDs::COLS] = colsRows.x;
    intValues[IntConstantIDs::ROWS] = colsRows.y;
    intValues[IntConstantIDs::DISTANCE_ORDER] = params.distanceOrder;
    intValues[IntConstantIDs::SCAN_METRIC] = params.metric;
    intValues[IntConstantIDs::FOCUS_METHOD] = params.method;
    intValues[IntConstantIDs::CLOSEST_VIEWS] = params.closestViews;
    intValues[IntConstantIDs::BLOCK_SAMPLING] = (params.blockSampling > 0.000000001);
    intValues[IntConstantIDs::YUV_DISTANCE] = params.colorDistance;
    intValues[IntConstantIDs::CLOCK_SEED] = std::clock();
    intValues[IntConstantIDs::BLEND_ADDRESS_MODE] = addressMode;
    cudaMemcpyToSymbol(Kernels::Constants::intConstants, intValues.data(), intValues.size() * sizeof(int));
    
    std::vector<float> floatValues(FloatConstantIDs::FLOAT_CONSTANTS_COUNT);
    floatValues[FloatConstantIDs::SPACE] = params.space;
    float rangeSize = params.scanRange.y - params.scanRange.x; 
    floatValues[FloatConstantIDs::DESCENT_START_STEP] = rangeSize/DESCENT_START_POINTS/4.0f;
    floatValues[FloatConstantIDs::SCAN_RANGE_SIZE] = rangeSize;
    floatValues[FloatConstantIDs::SCAN_RANGE_START] = params.scanRange.x;
    floatValues[FloatConstantIDs::SCAN_RANGE_END] = params.scanRange.y;
    float pyramidBroadStep = rangeSize/PYRAMID_DIVISIONS_BROAD;
    floatValues[FloatConstantIDs::PYRAMID_BROAD_STEP] = pyramidBroadStep; 
    floatValues[FloatConstantIDs::PYRAMID_NARROW_STEP] = pyramidBroadStep/PYRAMID_DIVISIONS_NARROW; 
    floatValues[FloatConstantIDs::DOF_DISTANCE] = params.dofDistWidthMax.x;
    floatValues[FloatConstantIDs::DOF_WIDTH] = params.dofDistWidthMax.y;
    floatValues[FloatConstantIDs::DOF_MAX] = params.dofDistWidthMax.z;
    floatValues[FloatConstantIDs::MIST_START] = params.mistStartEndCol.x;
    floatValues[FloatConstantIDs::MIST_END] = params.mistStartEndCol.y;
    floatValues[FloatConstantIDs::MIST_COLOR] = params.mistStartEndCol.z;
    floatValues[FloatConstantIDs::FOCUS_METHOD_PARAMETER] = params.methodParameter;
    float2 pixelSize{1.0f/resolution.x, 1.0f/resolution.y}; 
    floatValues[FloatConstantIDs::PX_SIZE_X] = pixelSize.x;
    floatValues[FloatConstantIDs::PX_SIZE_Y] = pixelSize.y;
    cudaMemcpyToSymbol(Kernels::Constants::floatConstants, floatValues.data(), floatValues.size() * sizeof(float));

    std::vector<void*> dataPointers(DataPointersIDs::POINTERS_COUNT);
    dataPointers[DataPointersIDs::SURFACES] = reinterpret_cast<void*>(surfaceObjectsArr);
    dataPointers[DataPointersIDs::TEXTURES] = reinterpret_cast<void*>(textureObjectsArr);
    dataPointers[DataPointersIDs::SECONDARY_TEXTURES] = reinterpret_cast<void*>(secondaryTextureObjectsArr);
    dataPointers[DataPointersIDs::MIP_TEXTURES] = reinterpret_cast<void*>(mipTextureObjectsArr);
    cudaMemcpyToSymbol(Kernels::Constants::dataPointers, dataPointers.data(), dataPointers.size() * sizeof(void*));
            
    constexpr float START_STEP{1.0f/(DESCENT_START_POINTS)};
    float startFocuses[DESCENT_START_POINTS];
    for(int i=1; i<DESCENT_START_POINTS; i++)
        startFocuses[i] = params.scanRange.x+START_STEP*i*rangeSize;
    cudaMemcpyToSymbol(Kernels::Constants::descentStartPoints, startFocuses, DESCENT_START_POINTS * sizeof(float));
    
    float hierarchySteps[HIERARCHY_DIVISIONS];
    int hierarchySamplings[HIERARCHY_DIVISIONS];
    int currentSampling{15};
    float currentRange{rangeSize};
    for(int i=0; i<HIERARCHY_DIVISIONS; i++)
    {
        hierarchySamplings[i] = currentSampling;
        hierarchySteps[i] = params.scanRange.x+currentRange/currentSampling;
        currentSampling /= 2;
        currentRange /= 2;
    }
    cudaMemcpyToSymbol(Kernels::Constants::hierarchySteps, hierarchySteps, HIERARCHY_DIVISIONS * sizeof(float));
    cudaMemcpyToSymbol(Kernels::Constants::hierarchySamplings, hierarchySamplings, HIERARCHY_DIVISIONS * sizeof(int));
  
    float2 pixelSizeBlock{params.blockSampling*pixelSize.x, params.blockSampling*pixelSize.y}; 
    std::vector<float2> blockOffsets{ {0.0f, 0.0f}, {-1.0f*pixelSizeBlock.x, 0.5f*pixelSizeBlock.y}, {0.5f*pixelSizeBlock.x, 1.0f*pixelSizeBlock.y}, {1.0f*pixelSizeBlock.x, -0.5f*pixelSizeBlock.y}, {-0.5f*pixelSizeBlock.x, -1.0f*pixelSizeBlock.y} };
    cudaMemcpyToSymbol(Kernels::Constants::blockOffsets, blockOffsets.data(), BLOCK_OFFSET_COUNT * sizeof(float2));
}

void Interpolator::loadGPUOffsets(glm::vec2 viewCoordinates)
{
    float aspect = (static_cast<float>(resolution.x)/resolution.y) / inputCamerasSpacingAspect;
    std::vector<float2> offsets(colsRows.x*colsRows.y);
    int gridID = 0; 
    for(int row=0; row<colsRows.y; row++) 
    {     
        gridID = row*colsRows.x;
        for(int col=0; col<colsRows.x; col++) 
        {
            float2 offset{(viewCoordinates.x-col)/colsRows.x, (viewCoordinates.y-row)/colsRows.y};
            offset.y *= aspect;
            offsets[gridID] = offset;
            gridID++;
        }
    }
    cudaMemcpyToSymbol(Kernels::Constants::offsets, offsets.data(), offsets.size() * sizeof(float2));
}

std::vector<float> Interpolator::generateWeights(glm::vec2 coords)
{
    auto maxDistance = glm::distance(glm::vec2(0,0), glm::vec2(colsRows));
    float weightSum{0};
    std::vector<float> weightVals;
    for(int row=0; row<colsRows.y; row++) 
        for(int col=0; col<colsRows.x; col++) 
        {
            float weight = maxDistance - glm::distance(coords, glm::vec2(col, row));
            weightSum += weight;
            weightVals.push_back(weight);
        }
    for(auto &weight : weightVals)
        weight /= weightSum; 
    return weightVals;
}

Interpolator::ClosestViews Interpolator::selectNeighboringFrames(int count, std::vector<float> &weights)
{
    std::vector<int> indices(weights.size());
    std::iota(indices.begin(), indices.end(), 0);
    auto comparator = [&weights](int a, int b){ return weights[a] > weights[b]; };
    std::sort(indices.begin(), indices.end(), comparator);

    std::vector<float> closestWeights(count);
    std::vector<int> closestIds(count);
    float totalWeights{0};
    for(int i=0; i<count; i++)
    {
        closestIds[i] = indices[i];
        closestWeights[i] = weights[i];
        totalWeights += weights[i];
    }
    for(auto &w : closestWeights)
        w /= totalWeights;
    return {closestWeights, closestIds};
}

void Interpolator::loadGPUWeights(glm::vec2 viewCoordinates)
{
    auto weights = generateWeights(viewCoordinates);
    cudaMemcpyToSymbol(Kernels::Constants::weights, weights.data(), weights.size() * sizeof(float));
    auto closest = selectNeighboringFrames(NEIGHBOR_VIEWS_COUNT, weights);
    cudaMemcpyToSymbol(Kernels::Constants::neighborWeights, closest.weights.data(), closest.weights.size() * sizeof(float));
    cudaMemcpyToSymbol(Kernels::Constants::neighborIds, closest.ids.data(), closest.ids.size() * sizeof(int));
}

std::vector<std::string> Interpolator::InterpolationParams::split(std::string input, char delimiter) const
{
    std::vector<std::string> values;
    std::stringstream ss(input); 
    std::string value; 
    while(getline(ss, value, delimiter))
        values.push_back(value);
    return values;
}

glm::vec3 Interpolator::InterpolationParams::parseCoordinates(std::string coordinates) const
{
    auto values = split(coordinates);
    glm::vec3 numbers{0};
    int i{0};
    for(auto& value : values)
    {
        if(i>3)
            throw std::runtime_error("Coordinates too long!");
        if(value[0] == '#')
        {
            value.erase(0,1);
            auto newValue = "0x"+value;
            numbers[i] = std::stoul(newValue, nullptr, 16); 
        }
        else
            numbers[i] = std::stof(value);
        i++;
    }

    return numbers;
}

void Interpolator::prepareClosestFrames(glm::vec2 viewCoordinates)
{
    constexpr int CLOSEST_FRAMES_COUNT{4};
    
    glm::ivec2 downCoords{glm::floor(viewCoordinates)};
    glm::ivec2 upCoords{glm::ceil(viewCoordinates)};
    
    std::vector<float> closestFramesWeights(CLOSEST_FRAMES_COUNT);
    std::vector<glm::ivec2> closestFramesCoords(CLOSEST_FRAMES_COUNT);
    glm::vec2 unitPos{glm::fract(viewCoordinates)};

    closestFramesCoords[ClosestFrames::TOP_LEFT] = {downCoords};
    closestFramesWeights[ClosestFrames::TOP_LEFT] = (1 - unitPos.x) * (1 - unitPos.y);
    
    closestFramesCoords[ClosestFrames::TOP_RIGHT] = {upCoords.x, downCoords.y};;
    closestFramesWeights[ClosestFrames::TOP_RIGHT] = unitPos.x * (1 - unitPos.y);
    
    closestFramesCoords[ClosestFrames::BOTTOM_LEFT] = {downCoords.x, upCoords.y};
    closestFramesWeights[ClosestFrames::BOTTOM_LEFT] = (1 - unitPos.x) * unitPos.y;

    closestFramesCoords[ClosestFrames::BOTTOM_RIGHT] = {upCoords};
    closestFramesWeights[ClosestFrames::BOTTOM_RIGHT] = unitPos.x * unitPos.y;
  
    std::vector<int> closestFramesCoordsLinear;
    for(auto const &coords : closestFramesCoords)
       closestFramesCoordsLinear.push_back(coords.y*colsRows.x+coords.x); 
     
    cudaMemcpyToSymbol(Kernels::Constants::closestCoords, closestFramesCoordsLinear.data(), closestFramesCoordsLinear.size() * sizeof(int));
    cudaMemcpyToSymbol(Kernels::Constants::closestWeights, closestFramesWeights.data(), closestFramesWeights.size() * sizeof(float));    
}

MapFilter Interpolator::InterpolationParams::parseMapFilter(std::string filter) const
{
    if(filter == "NONE")
        return MapFilter::NONE;
    else if(filter == "MED")
        return MapFilter::MEDIAN;
    else if(filter == "SNN")
        return MapFilter::SNN;
    else if(filter == "KUW")
        return MapFilter::KUWAHARA;
    std::cerr << "Focus map filter set to default." << std::endl;
    return MapFilter::NONE;
}

ColorDistance Interpolator::InterpolationParams::parseColorDistance(std::string distance) const
{
    if(distance == "RGB")
        return ColorDistance::RGB;
    else if(distance == "Y")
        return ColorDistance::Y;
    else if(distance == "YUV")
        return ColorDistance::YUV;
    else if(distance == "YUVw")
        return ColorDistance::YUVw;
    std::cerr << "Color metric distance metric set to default." << std::endl;
    return ColorDistance::RGB;
}

ScanMetric Interpolator::InterpolationParams::parseMetric(std::string metric) const
{
    if(metric == "VAR")
        return ScanMetric::VARIANCE;
    else if(metric == "RANGE")
        return ScanMetric::RANGE;
    else if(metric == "ERANGE")
        return ScanMetric::ELEMENT_RANGE;
    else if(metric == "MAD")
        return ScanMetric::MAD;
    std::cerr << "Scan metric set to default." << std::endl;
    return ScanMetric::VARIANCE;
}

AddressMode Interpolator::parseAddressMode(std::string addressMode) const
{
    if(addressMode == "WRAP")
        return AddressMode::WRAP;
    else if(addressMode == "CLAMP")
        return AddressMode::CLAMP;
    else if(addressMode == "BORDER")
        return AddressMode::BORDER;
    else if(addressMode == "MIRROR")
        return AddressMode::MIRROR;
    else if(addressMode == "BLEND")
        return AddressMode::BLEND;
    else if(addressMode == "ALTER")
        return AddressMode::ALTER;
    std::cerr << "Texture address mode set to default." << std::endl;
    return AddressMode::CLAMP;
}
 
FocusMethod Interpolator::InterpolationParams::parseMethod(std::string method) const
{
    if(method == "OD")
        return FocusMethod::ONE_DISTANCE;
    else if(method == "BF")
        return FocusMethod::BRUTE_FORCE;
    else if(method == "BFET")
        return FocusMethod::BRUTE_FORCE_EARLY;
    else if(method == "VS")
        return FocusMethod::VARIABLE_SCAN;
    else if(method == "VSET")
        return FocusMethod::VARIABLE_SCAN_EARLY;
    else if(method == "TD")
        return FocusMethod::TOP_DOWN;
    else if(method == "RAND")
        return FocusMethod::RANDOM;
    else if(method == "HIER")
        return FocusMethod::HIERARCHY;
    else if(method == "PYR")
        return FocusMethod::PYRAMID;
    else if(method == "DESC")
        return FocusMethod::DESCENT;
    std::cerr << "Scan method set to default." << std::endl;
    return FocusMethod::BRUTE_FORCE;
} 

void Interpolator::runKernel(KernelType type, KernelParams params)
{
    dim3 dimBlock(16, 16, 1);
    switch(type)
    {
        case PROCESS:
        {
            dim3 dimGrid(glm::ceil(static_cast<float>(resolution.x)/dimBlock.x), glm::ceil(static_cast<float>(resolution.y)/dimBlock.y), 1);
            Kernels::process<<<dimGrid, dimBlock, sharedSize>>>();
        }
        break;

        case ARR_RGB_YUV:
        {
            dim3 dimGrid(glm::ceil(static_cast<float>(params.width)/dimBlock.x), glm::ceil(static_cast<float>(params.height)/dimBlock.y), 1);
            Kernels::Conversion::RGBtoYUV<<<dimGrid, dimBlock, sharedSize>>>(params.data, params.width, params.height);
        }
        break;

        case ARR_YUV_RGB:
        {
            dim3 dimGrid(glm::ceil(static_cast<float>(params.width)/dimBlock.x), glm::ceil(static_cast<float>(params.height)/dimBlock.y), 1);
            Kernels::Conversion::YUVtoRGB<<<dimGrid, dimBlock, sharedSize>>>(params.data, params.width, params.height);
        }
        break;
        
        case POST:
        {
            dim3 dimGrid(glm::ceil(static_cast<float>(resolution.x)/dimBlock.x), glm::ceil(static_cast<float>(resolution.y)/dimBlock.y), 1);
            Kernels::PostProcess::applyEffects<<<dimGrid, dimBlock, sharedSize>>>();
        }
        break;
        
        case FILTER:
        {
            dim3 dimGrid(glm::ceil(static_cast<float>(resolution.x)/dimBlock.x), glm::ceil(static_cast<float>(resolution.y)/dimBlock.y), 1);
            Kernels::PostProcess::filterMap<<<dimGrid, dimBlock, sharedSize>>>(params.filter, secondMapActive, params.firstFilter);
        }
        break;
    }
    cudaDeviceSynchronize();
}

void Interpolator::testKernel(KernelType kernel, std::string label, int runs, std::vector<MapFilter> filters)
{
    std::cout << "Elapsed time of "+label+": "<<std::endl;
    float avgTime{0};
    for(int i=0; i<runs; i++)
    {
        Timer timer;
        
        if(kernel == FILTER)
        {
            bool firstFilter{true};
            for(const auto& filter : filters)
            {
                KernelParams params;
                params.filter = filter;
                params.firstFilter = firstFilter;
                runKernel(kernel, params);
                firstFilter = false;
                secondMapActive = !secondMapActive;
            }
        }
        else 
            runKernel(kernel);
        auto time = timer.stop();
        avgTime += time;
        std::cout << "Run #" << i<< ": " << time << " ms" << std::endl;
    }
    std::cout << "Average of " << runs << " runs of "+label+": " << avgTime/runs << " ms" << std::endl;
}

void Interpolator::interpolate(InterpolationParams params)
{
    glm::vec2 coords = glm::vec2(colsRows-1)*params.coordinates;
    loadGPUWeights(coords);
    prepareClosestFrames(coords);
    loadGPUOffsets(coords);   
    loadGPUConstants(params);
    
    testKernel(PROCESS, "interpolation", params.runs);
    testKernel(FILTER, "map filtering", params.runs, params.mapFilters);
    testKernel(POST, "post process", params.runs);

    storeResults(params.outputPath);
}

void Interpolator::storeResults(std::string path)
{
    std::cout << "Storing results..." << std::endl;
    LoadingBar bar(FileNames::OUTPUT_COUNT-1);
    std::vector<uint8_t> data(resolution.x*resolution.y*resolution.z, 255);

    size_t pitch = resolution.x*resolution.z;
    void *imageData;
    cudaMalloc(&imageData, resolution.x*resolution.y*resolution.z);

    for(int i=0; i<FileNames::OUTPUT_COUNT; i++) 
    {
        cudaMemcpy2DFromArray(imageData, pitch, reinterpret_cast<cudaArray*>(surfaceOutputArrays[i]), 0, 0, pitch, resolution.y, cudaMemcpyDeviceToDevice);
        if(i > FileNames::FOCUS_MAP_POST_SECOND && useYUV)
            runKernel(ARR_YUV_RGB, {imageData, resolution.x, resolution.y});

        cudaMemcpy2D(data.data(), pitch, imageData, pitch, pitch, resolution.y, cudaMemcpyDeviceToHost);
        if(i <= FileNames::FOCUS_MAP_POST_SECOND)
        {
            if((secondMapActive && i == FOCUS_MAP_POST) || (!secondMapActive && i == FOCUS_MAP_POST_SECOND) )
                continue;
            stbi_write_hdr((std::filesystem::path(path)/(fileNames[i]+".hdr")).c_str(), resolution.x, resolution.y, 1, reinterpret_cast<float*>(data.data()));
        }
        else
            stbi_write_png((std::filesystem::path(path)/(fileNames[i]+".png")).c_str(), resolution.x, resolution.y, resolution.z, data.data(), resolution.x*resolution.z);
        bar.add();
    }
    cudaFree(imageData);
}
