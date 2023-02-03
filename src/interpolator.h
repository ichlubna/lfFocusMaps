#include <vector>
#include <glm/glm.hpp>
#include <string>

class Interpolator
{
    public:
    Interpolator(std::string inputPath);
    ~Interpolator();
    void interpolate(std::string outputPath, std::string coordinates, std::string method, int runs);

    private:
    enum Method {BRUTE_FORCE};
    static constexpr int OUTPUT_SURFACE_COUNT{2};
    std::vector<int*> surfaceInputArrays;
    std::vector<int*> surfaceOutputArrays;
    void *surfaceObjectsArr;
    void *textureObjectsArr;
    enum FileNames {FOCUS_MAP=0, RENDER_IMAGE=1};
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
    void loadGPUConstants();
    void loadGPUWeights(glm::vec2 viewCoordinates);
    int* loadImageToArray(const uint8_t *data, glm::ivec3 size);
    Interpolator::Method parseMethod(std::string method);
    glm::vec2 parseCoordinates(std::string coordinates);
    void storeResults(std::string path);
    std::vector<float> generateWeights(glm::vec2 coords);
    std::pair<int, int*> createSurfaceObject(glm::ivec3 size, const uint8_t *data=nullptr);
    int createTextureObject(const uint8_t *data, glm::ivec3 size);
    void prepareClosestFrames(glm::vec2 viewCoordinates);
};
