#include <stdexcept>
#include <vector>
#include <glm/glm.hpp>
#include <string>
#include "methods.h"

class Interpolator
{
    public:
    class InterpolationParams
    {
        public:
        InterpolationParams* setOutputPath(std::string path)
        {
            outputPath = path; 
            return this;
        }
        InterpolationParams* setCoordinates(std::string inputCoordinates)
        {
            coordinates = parseCoordinates(inputCoordinates); 
            return this;
        }
        
        InterpolationParams* setMethod(std::string inputMethod)
        {
            method = parseMethod(inputMethod);
            return this;
        }
         
        InterpolationParams* setMetric(std::string inputMetric)
        {
            metric = parseMetric(inputMetric);
            return this;
        }
        
        InterpolationParams* setSpace(float inputSpace)
        {
            if(inputSpace == 0)
                space = 1;
            else
                space = inputSpace;
            return this;
        }
        
        InterpolationParams* setMethodParameter(float parameter)
        {
            methodParameter = parameter; 
            return this;
        }
        
        InterpolationParams* setClosestViews(bool closest=true)
        {
            closestViews = closest;
            return this;
        }
        
        InterpolationParams* setBlockSampling(bool block=true)
        {
            blockSampling = block;
            return this;
        }
        
        InterpolationParams* setNoMap(bool no=true)
        {
            noMap = no;
            return this;
        }
        
        InterpolationParams* setColorDistance(std::string distance)
        {
            colorDistance = parseColorDistance(distance);
            return this;
        }

        InterpolationParams* setScanRange(float range)
        {
            if(range < 0)
                throw std::runtime_error("Focusing scanning range cannot be negative!");
            scanRange = range;
            return this;
        }
        
        InterpolationParams* setRuns(int runsCount)
        {
            if(runsCount < 0)
                throw std::runtime_error("Number of kernel runs cannot be negative!");
            else if(runsCount > 0)
                runs = runsCount;
            return this;
        }

        InterpolationParams* setDistanceOrder(int order)
        {
            if(order < 0)
                throw std::runtime_error("Distance function order cannot be negative!");
            else if(order > 0)
                distanceOrder = order;
            return this;
        }

        InterpolationParams* set()
        {
        
            return this;
        }
        
        std::string outputPath;
        glm::vec2 coordinates;
        FocusMethod method;
        float space;
        ScanMetric metric;
        float methodParameter;
        bool closestViews{false};
        bool blockSampling{false};
        bool noMap{false};
        ColorDistance colorDistance;
        float scanRange;
        int distanceOrder{1};
        int runs{1};
        
        private:
        glm::vec2 parseCoordinates(std::string coordinates) const;
        FocusMethod parseMethod(std::string inputMethod) const;
        ScanMetric parseMetric(std::string inputMetric) const;
        ColorDistance parseColorDistance(std::string distance) const;
    };

    Interpolator(std::string inputPath, std::string mode, bool useSecondary, bool mips, bool yuv, bool useAspect);
    ~Interpolator();
    void interpolate(InterpolationParams params);

    private:
    class TexturesInfo
    {
        public:
        glm::ivec2 colsRows;
        glm::ivec3 resolution;
    };
    enum KernelType{PROCESS, ARR_RGB_YUV, ARR_YUV_RGB};
    class KernelParams
    {
        public:
        void *data;
        int width;
        int height;
    };
    void runKernel(KernelType, KernelParams={});

    static constexpr int OUTPUT_SURFACE_COUNT{2};
    AddressMode addressMode{CLAMP};
    bool useSecondaryFolder{false};
    bool useMips{false};
    bool useYUV{false};
    bool useAspect{false};
    std::vector<int*> surfaceInputArrays;
    std::vector<int*> surfaceOutputArrays;
    void *surfaceObjectsArr;
    void *textureObjectsArr;
    void *secondaryTextureObjectsArr;
    void *mipTextureObjectsArr;
    const std::vector<std::string> fileNames{"focusMap", "renderImage"};
    float *weightsGPU;
    float *closestFramesWeightsGPU;
    int *closestFramesCoordsLinearGPU;
    size_t channels{4};
    size_t sharedSize{0};
    glm::ivec2 colsRows;
    glm::ivec3 resolution;
    std::string input;
    void init();
    void loadGPUOffsets(glm::vec2 viewCoordinates);
    void loadGPUData();
    TexturesInfo loadTextures(std::string input, void **textures);
    void loadGPUConstants(InterpolationParams params);
    void loadGPUWeights(glm::vec2 viewCoordinates);
    int* loadImageToArray(const uint8_t *data, glm::ivec3 size);
    void storeResults(std::string path, bool noMap);
    std::vector<float> generateWeights(glm::vec2 coords);
    std::pair<int, int*> createSurfaceObject(glm::ivec3 size, const uint8_t *data=nullptr);
    int createTextureObject(const uint8_t *data, glm::ivec3 size);
    void prepareClosestFrames(glm::vec2 viewCoordinates);
    AddressMode parseAddressMode(std::string addressMode) const;
};
