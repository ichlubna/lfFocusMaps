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

Interpolator::Interpolator(std::string inputPath) : input{inputPath}
{
    init();
}

Interpolator::~Interpolator()
{
    cudaDeviceReset();
}

void Interpolator::init()
{
    loadGPUData();
    loadGPUConstants();
    sharedSize = 0;//sizeof(half)*colsRows.x*colsRows.y;
}

int Interpolator::createTextureObject(const uint8_t *data, glm::ivec3 size)
{
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
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;
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

void Interpolator::loadGPUData()
{
    LfLoader lfLoader;
    lfLoader.loadData(input);
    colsRows = lfLoader.getColsRows();
    resolution = lfLoader.imageResolution();

    std::cout << "Uploading data to GPU..." << std::endl;
    LoadingBar bar(lfLoader.imageCount()+OUTPUT_SURFACE_COUNT);

    /*
    std::vector<cudaTextureObject_t> textures;
    for(int col=0; col<colsRows.x; col++)
        for(int row=0; row<colsRows.y; row++)
        { 
            textures.push_back(createTextureObject(lfLoader.image({col, row}).data(), resolution)); 
            bar.add();
        }

    cudaMalloc(&textureObjectsArr, textures.size()*sizeof(cudaTextureObject_t));
    cudaMemcpy(textureObjectsArr, textures.data(), textures.size()*sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    */

    
    std::vector<cudaSurfaceObject_t> surfaces;
    for(int col=0; col<colsRows.x; col++)
        for(int row=0; row<colsRows.y; row++)
        {
            auto surface = createSurfaceObject(resolution, lfLoader.image({col, row}).data());
            surfaces.push_back(surface.first);  
            surfaceInputArrays.push_back(surface.second);
            bar.add();
        }

    for(int i=0; i<OUTPUT_SURFACE_COUNT; i++)
    {
        auto surface = createSurfaceObject(resolution);
        surfaces.push_back(surface.first);  
        surfaceOutputArrays.push_back(surface.second);
        bar.add();
    }
    cudaMalloc(&surfaceObjectsArr, surfaces.size()*sizeof(cudaTextureObject_t));
    cudaMemcpy(surfaceObjectsArr, surfaces.data(), surfaces.size()*sizeof(cudaSurfaceObject_t), cudaMemcpyHostToDevice);
}

void Interpolator::loadGPUConstants()
{
    std::vector<int> values{resolution.x, resolution.y, colsRows.x, colsRows.y, colsRows.x*colsRows.y, FileNames::FOCUS_MAP, FileNames::RENDER_IMAGE};
    cudaMemcpyToSymbol(Kernels::constants, values.data(), values.size() * sizeof(int));
}

void Interpolator::loadGPUOffsets(glm::vec2 viewCoordinates)
{
    std::vector<float2> offsets(colsRows.x*colsRows.y);
    for(int col=0; col<colsRows.x; col++)
        for(int row=0; row<colsRows.y; row++)
        {
            int gridID = row*colsRows.x + col; 
            float2 offset{(col-viewCoordinates.x)/colsRows.x, (row-viewCoordinates.y)/colsRows.y};
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

glm::vec2 Interpolator::parseCoordinates(std::string coordinates)
{
    constexpr char delim{'_'};
    std::vector <std::string> numbers;
    std::stringstream ss(coordinates); 
    std::string value; 
    while(getline(ss, value, delim))
        numbers.push_back(value);

    glm::vec2 coords;
    int i{0};
    for (const auto &number : numbers)
    {
        float value = std::stof(number);
        coords[i] = value*(colsRows[i%2]-1);
        i++;
    }
    return coords;
}

void Interpolator::prepareClosestFrames(glm::vec2 viewCoordinates)
{
    constexpr int CLOSEST_FRAMES_COUNT{4};
    
    glm::ivec2 downCoords{glm::floor(viewCoordinates)};
    glm::ivec2 upCoords{glm::ceil(viewCoordinates)};
    
    std::vector<float> closestFramesWeights(CLOSEST_FRAMES_COUNT);
    std::vector<glm::ivec2> closestFramesCoords(CLOSEST_FRAMES_COUNT);
    glm::vec2 unitPos{glm::fract(viewCoordinates)};

    closestFramesCoords[Kernels::TOP_LEFT] = {downCoords};
    closestFramesWeights[Kernels::TOP_LEFT] = (1 - unitPos.x) * (1 - unitPos.y);
    
    closestFramesCoords[Kernels::TOP_RIGHT] = {upCoords.x, downCoords.y};;
    closestFramesWeights[Kernels::TOP_RIGHT] = unitPos.x * (1 - unitPos.y);
    
    closestFramesCoords[Kernels::BOTTOM_LEFT] = {downCoords.x, upCoords.y};
    closestFramesWeights[Kernels::BOTTOM_LEFT] = (1 - unitPos.x) * unitPos.y;

    closestFramesCoords[Kernels::BOTTOM_RIGHT] = {upCoords};
    closestFramesWeights[Kernels::BOTTOM_RIGHT] = unitPos.x * unitPos.y;
  
    std::vector<int> closestFramesCoordsLinear;
    for(auto const &coords : closestFramesCoords)
       closestFramesCoordsLinear.push_back(coords.y*colsRows.x+coords.x); 
     
    cudaMalloc(reinterpret_cast<void **>(&closestFramesCoordsLinearGPU), sizeof(int)*closestFramesCoordsLinear.size());
    cudaMemcpy(closestFramesCoordsLinearGPU, closestFramesCoordsLinear.data(), closestFramesCoordsLinear.size()*sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(reinterpret_cast<void **>(&closestFramesWeightsGPU), sizeof(float)*closestFramesWeights.size());
    cudaMemcpy(closestFramesWeightsGPU, closestFramesWeights.data(), closestFramesWeights.size()*sizeof(float), cudaMemcpyHostToDevice);
}

Kernels::FocusMethod parseMethod(std::string method)
{
    if(method == "OD")
        return Kernels::FocusMethod::ONE_DISTANCE;
    else if(method == "BF")
        return Kernels::FocusMethod::BRUTE_FORCE;
    return Kernels::FocusMethod::BRUTE_FORCE;
} 

void Interpolator::interpolate(std::string outputPath, std::string coordinates, std::string method, float methodParameter, bool closestViews, bool blockSampling, int inputRange, int runs)
{
    glm::vec2 coords = parseCoordinates(coordinates);
    loadGPUWeights(coords);
    prepareClosestFrames(coords);
    loadGPUOffsets(coords);   
    
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(resolution.x/dimBlock.x, resolution.y/dimBlock.y, 1);

    int range = (inputRange > 0) ? inputRange : resolution.x/2;
    auto focusMethod = parseMethod(method); 

    std::cout << "Elapsed time: "<<std::endl;
    float avgTime{0};
    for(int i=0; i<runs; i++)
    {
        Timer timer;
        Kernels::process<<<dimGrid, dimBlock, sharedSize>>>
        (focusMethod, methodParameter, closestViews, blockSampling, 
        reinterpret_cast<cudaTextureObject_t*>(textureObjectsArr), reinterpret_cast<cudaSurfaceObject_t*>(surfaceObjectsArr),
        weightsGPU, closestFramesWeightsGPU, closestFramesCoordsLinearGPU, range);
        auto time = timer.stop();
        avgTime += time;
        std::cout << "Run #" << i<< ": " << time << " ms" << std::endl;
    }
    std::cout << "Average of " << runs << " runs: " << avgTime/runs << " ms" << std::endl;

    storeResults(outputPath);
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
