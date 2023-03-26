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
        
        InterpolationParams* setDof(std::string distWidth)
        {
            dofDistWidthMax = parseCoordinates(distWidth); 
            return this;
        }
        
        InterpolationParams* setMist(std::string mist)
        {
            mistStartEndCol = parseCoordinates(mist); 
            return this;
        }
        
        InterpolationParams* setMethod(std::string inputMethod)
        {
            method = parseMethod(inputMethod);
            return this;
        }
        
        InterpolationParams* setMapFilter(std::string filter)
        {
            mapFilter = parseMapFilter(filter);
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
        
        InterpolationParams* setBlockSampling(float block)
        {
            blockSampling = block;
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
        float blockSampling{0};
        glm::vec3 dofDistWidthMax{0,0,0};
        glm::vec3 mistStartEndCol{0,0,0};
        ColorDistance colorDistance;
        MapFilter mapFilter;
        float scanRange;
        int distanceOrder{1};
        int runs{1};
        
        private:
        glm::vec3 parseCoordinates(std::string coordinates) const;
        FocusMethod parseMethod(std::string inputMethod) const;
        ScanMetric parseMetric(std::string inputMetric) const;
        ColorDistance parseColorDistance(std::string distance) const;
        MapFilter parseMapFilter(std::string filter) const;
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
    enum KernelType{PROCESS, ARR_RGB_YUV, ARR_YUV_RGB, POST};
    class KernelParams
    {
        public:
        void *data;
        int width;
        int height;
    };
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
    const std::vector<std::string> fileNames{"focusMap", "focusMapPost", "renderImage", "renderImagePost", "renderImagePostFiltered"};
    float *weightsGPU;
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
    int* loadImageToArray(const uint8_t *data, glm::ivec3 size, bool copyFromDevice);
    void storeResults(std::string path);
    std::vector<float> generateWeights(glm::vec2 coords);
    std::pair<int, int*> createSurfaceObject(glm::ivec3 size, const uint8_t *data=nullptr, bool copyFromDevice=false);
    int createTextureObject(const uint8_t *data, glm::ivec3 size);
    void prepareClosestFrames(glm::vec2 viewCoordinates);
    void runKernel(KernelType, KernelParams={});
    void testKernel(KernelType kernel, std::string label, int runs);
    AddressMode parseAddressMode(std::string addressMode) const;
};
