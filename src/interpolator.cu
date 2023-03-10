#define GLM_FORCE_SWIZZLE
#include <sstream>
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

Interpolator::Interpolator(std::string inputPath, std::string mode, bool useSecondary, bool mips) : useSecondaryFolder{useSecondary}, useMips{mips}, input{inputPath}
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
        cudaAddressMode = cudaAddressModeClamp;
    break;
    }

    cudaChannelFormatDesc channels = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray *arr;
    cudaMallocArray(&arr, &channels, size.x, size.y);
    cudaMemcpy2DToArray(arr, 0, 0, data, size.x*size.z, size.x*size.z, size.y, cudaMemcpyHostToDevice);
    
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = arr;
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.filterMode = cudaFilterModeLinear;
    //cudaAddressMode = cudaAddressModeBorder;
    texDescr.normalizedCoords = true;
    texDescr.addressMode[0] = cudaAddressMode;
    texDescr.addressMode[1] = cudaAddressMode;
    texDescr.readMode = cudaReadModeNormalizedFloat;
    cudaTextureObject_t texObj{0};
    cudaCreateTextureObject(&texObj, &texRes, &texDescr, NULL);
    return texObj;
}

std::pair<int, int*> Interpolator::createSurfaceObject(glm::ivec3 size, const uint8_t *data)
{
    auto arr = loadImageToArray(data, size);
    cudaResourceDesc surfRes;
    memset(&surfRes, 0, sizeof(cudaResourceDesc));
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = reinterpret_cast<cudaArray*>(arr);
    cudaSurfaceObject_t surfObj = 0;
    cudaCreateSurfaceObject(&surfObj, &surfRes);
    return {surfObj, arr};
}

int* Interpolator::loadImageToArray(const uint8_t *data, glm::ivec3 size)
{
    cudaChannelFormatDesc channels = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned); 
    cudaArray *arr;
    cudaMallocArray(&arr, &channels, size.x, size.y, cudaArraySurfaceLoadStore);
    if(data != nullptr)
        cudaMemcpy2DToArray(arr, 0, 0, data, size.x*size.z, size.x*size.z, size.y, cudaMemcpyHostToDevice);
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
    for(int i=0; i<OUTPUT_SURFACE_COUNT; i++)
    {
        auto surface = createSurfaceObject(resolution);
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
    intValues[IntConstantIDs::BLOCK_SAMPLING] = params.blockSampling;
    intValues[IntConstantIDs::YUV_DISTANCE] = params.YUVDistance;
    intValues[IntConstantIDs::CLOCK_SEED] = std::clock();
    intValues[IntConstantIDs::BLEND_ADDRESS_MODE] = (addressMode == AddressMode::BLEND);
    cudaMemcpyToSymbol(Kernels::Constants::intConstants, intValues.data(), intValues.size() * sizeof(int));
    
    std::vector<float> floatValues(FloatConstantIDs::FLOAT_CONSTANTS_COUNT);
    floatValues[FloatConstantIDs::SPACE] = params.space;
    float range = (params.scanRange > 0) ? params.scanRange : 0.5f; 
    floatValues[FloatConstantIDs::DESCENT_START_STEP] = (static_cast<float>(range)/DESCENT_START_POINTS)/4.0f;
    floatValues[FloatConstantIDs::SCAN_RANGE] = range;
    floatValues[FloatConstantIDs::FOCUS_METHOD_PARAMETER] = params.methodParameter;
    cudaMemcpyToSymbol(Kernels::Constants::floatConstants, floatValues.data(), floatValues.size() * sizeof(float));

    std::vector<void*> dataPointers(DataPointersIDs::POINTERS_COUNT);
    dataPointers[DataPointersIDs::SURFACES] = reinterpret_cast<void*>(surfaceObjectsArr);
    dataPointers[DataPointersIDs::TEXTURES] = reinterpret_cast<void*>(textureObjectsArr);
    dataPointers[DataPointersIDs::SECONDARY_TEXTURES] = reinterpret_cast<void*>(secondaryTextureObjectsArr);
    dataPointers[DataPointersIDs::MIP_TEXTURES] = reinterpret_cast<void*>(mipTextureObjectsArr);
    dataPointers[DataPointersIDs::WEIGHTS] = reinterpret_cast<void*>(weightsGPU);
    dataPointers[DataPointersIDs::CLOSEST_WEIGHTS] = reinterpret_cast<void*>(closestFramesWeightsGPU);
    dataPointers[DataPointersIDs::CLOSEST_COORDS] = reinterpret_cast<void*>(closestFramesCoordsLinearGPU);
    cudaMemcpyToSymbol(Kernels::Constants::dataPointers, dataPointers.data(), dataPointers.size() * sizeof(void*));
            
    constexpr float START_STEP{1.0f/(DESCENT_START_POINTS)};
    float startFocuses[DESCENT_START_POINTS];
    for(int i=1; i<DESCENT_START_POINTS; i++)
        startFocuses[i] = START_STEP*i*range;
    cudaMemcpyToSymbol(Kernels::Constants::descentStartPoints, startFocuses, DESCENT_START_POINTS * sizeof(float));
    
    float hierarchySteps[HIERARCHY_DIVISIONS];
    int hierarchySamplings[HIERARCHY_DIVISIONS];
    int currentSampling{15};
    float currentRange{static_cast<float>(range)};
    for(int i=0; i<HIERARCHY_DIVISIONS; i++)
    {
        hierarchySamplings[i] = currentSampling;
        hierarchySteps[i] = currentRange/currentSampling;
        currentSampling /= 2;
        currentRange /= 2;
    }
    cudaMemcpyToSymbol(Kernels::Constants::hierarchySteps, hierarchySteps, HIERARCHY_DIVISIONS * sizeof(float));
    cudaMemcpyToSymbol(Kernels::Constants::hierarchySamplings, hierarchySamplings, HIERARCHY_DIVISIONS * sizeof(int));
}

void Interpolator::loadGPUOffsets(glm::vec2 viewCoordinates)
{
    float aspect = static_cast<float>(resolution.x)/resolution.y;
    std::vector<float2> offsets(colsRows.x*colsRows.y);
    for(int col=0; col<colsRows.x; col++)
        for(int row=0; row<colsRows.y; row++)
        {
            int gridID = row*colsRows.x + col; 
            float2 offset{(col-viewCoordinates.x)/colsRows.x, (row-viewCoordinates.y)/colsRows.y};
            offset.y *= aspect;
            offsets[gridID] = offset;
        }
    cudaMemcpyToSymbol(Kernels::offsets, offsets.data(), offsets.size() * sizeof(float2));
}

std::vector<float> Interpolator::generateWeights(glm::vec2 coords)
{
    auto maxDistance = glm::distance(glm::vec2(0,0), glm::vec2(colsRows));
    float weightSum{0};
    std::vector<float> weightVals;
    for(int col=0; col<colsRows.x; col++)
        for(int row=0; row<colsRows.y; row++)
        {
            float weight = maxDistance - glm::distance(coords, glm::vec2(col, row));
            weightSum += weight;
            weightVals.push_back(weight);
        }
    for(auto &weight : weightVals)
        weight /= weightSum; 
    return weightVals;
}

void Interpolator::loadGPUWeights(glm::vec2 viewCoordinates)
{
    cudaMalloc(reinterpret_cast<void **>(&weightsGPU), sizeof(float)*colsRows.x*colsRows.y);
    auto weights = generateWeights(viewCoordinates);
    cudaMemcpy(weightsGPU, weights.data(), weights.size()*sizeof(float), cudaMemcpyHostToDevice);
}

glm::vec2 Interpolator::InterpolationParams::parseCoordinates(std::string coordinates)
{
    constexpr char delim{'_'};
    std::vector <float> numbers;
    std::stringstream ss(coordinates); 
    std::string value; 
    while(getline(ss, value, delim))
        numbers.push_back(std::stof(value));

    return {numbers[0], numbers[1]};
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
     
    cudaMalloc(reinterpret_cast<void **>(&closestFramesCoordsLinearGPU), sizeof(int)*closestFramesCoordsLinear.size());
    cudaMemcpy(closestFramesCoordsLinearGPU, closestFramesCoordsLinear.data(), closestFramesCoordsLinear.size()*sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(reinterpret_cast<void **>(&closestFramesWeightsGPU), sizeof(float)*closestFramesWeights.size());
    cudaMemcpy(closestFramesWeightsGPU, closestFramesWeights.data(), closestFramesWeights.size()*sizeof(float), cudaMemcpyHostToDevice);
}

ScanMetric Interpolator::InterpolationParams::parseMetric(std::string metric)
{
    if(metric == "VAR")
        return ScanMetric::VARIANCE;
    else if(metric == "RANGE")
        return ScanMetric::RANGE;
    else if(metric == "IQR")
        return ScanMetric::IQR;
    else if(metric == "MAD")
        return ScanMetric::MAD;
    std::cerr << "Scan metric set to default." << std::endl;
    return ScanMetric::VARIANCE;
}

AddressMode Interpolator::parseAddressMode(std::string addressMode)
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
    std::cerr << "Texture address mode set to default." << std::endl;
    return AddressMode::CLAMP;
}
 
FocusMethod Interpolator::InterpolationParams::parseMethod(std::string method)
{
    if(method == "OD")
        return FocusMethod::ONE_DISTANCE;
    else if(method == "BF")
        return FocusMethod::BRUTE_FORCE;
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

void Interpolator::interpolate(InterpolationParams params)
{
    glm::vec2 coords = glm::vec2(colsRows-1)*params.coordinates;
    loadGPUWeights(coords);
    prepareClosestFrames(coords);
    loadGPUOffsets(coords);   
    loadGPUConstants(params);
    
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(glm::ceil(static_cast<float>(resolution.x)/dimBlock.x), glm::ceil(static_cast<float>(resolution.y)/dimBlock.y), 1);

    std::cout << "Elapsed time: "<<std::endl;
    float avgTime{0};
    for(int i=0; i<params.runs; i++)
    {
        Timer timer;
        Kernels::process<<<dimGrid, dimBlock, sharedSize>>>();
        auto time = timer.stop();
        avgTime += time;
        std::cout << "Run #" << i<< ": " << time << " ms" << std::endl;
    }
    std::cout << "Average of " << params.runs << " runs: " << avgTime/params.runs << " ms" << std::endl;
    storeResults(params.outputPath);
}

void Interpolator::storeResults(std::string path)
{
    std::cout << "Storing results..." << std::endl;
    LoadingBar bar(OUTPUT_SURFACE_COUNT);
    std::vector<uint8_t> data(resolution.x*resolution.y*resolution.z, 255);

    for(int i=0; i<OUTPUT_SURFACE_COUNT; i++) 
    {
        cudaMemcpy2DFromArray(data.data(), resolution.x*resolution.z, reinterpret_cast<cudaArray*>(surfaceOutputArrays[i]), 0, 0, resolution.x*resolution.z, resolution.y, cudaMemcpyDeviceToHost);
        stbi_write_png((std::filesystem::path(path)/(fileNames[i]+".png")).c_str(), resolution.x, resolution.y, resolution.z, data.data(), resolution.x*resolution.z);
        bar.add();
    }
}
