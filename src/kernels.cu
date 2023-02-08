#include <glm/glm.hpp>
#include <cuda_fp16.h>

namespace Kernels
{
    enum FocusMethod{ONE_DISTANCE, BRUTE_FORCE};
    enum ClosestFrames{TOP_LEFT=0, TOP_RIGHT=1, BOTTOM_LEFT=2, BOTTOM_RIGHT=3};
    __device__ constexpr bool GUESS_HANDLES{false};

    __device__ constexpr int CHANNEL_COUNT{4};
    __device__ constexpr int CONSTANTS_COUNT{7};
    __constant__ int constants[CONSTANTS_COUNT];
    __device__ int2 imgRes(){return {constants[0], constants[1]};}
    __device__ int2 colsRows(){return{constants[2], constants[3]};}
    __device__ int gridSize(){return constants[4];}
    __device__ int focusMapID(){return constants[5];}
    __device__ int renderImageID(){return constants[6];}

    __device__ constexpr int MAX_IMAGES{256};
    __constant__ float2 offsets[MAX_IMAGES];
    extern __shared__ half localMemory[];

    template <typename TT>
    class LocalArray
    {
        public:
        __device__ LocalArray(TT* inData) : data{inData}{}; 
        __device__ TT* ptr(int index)
        {
            return data+index;
        }
        
        template <typename T>
        __device__ T* ptr(int index) 
        {
            return reinterpret_cast<T*>(ptr(index));
        }  

        __device__ TT& ref(int index)
        {
            return *ptr(index);
        }
        
        template <typename T>
        __device__ T& ref(int index)
        {
            return *ptr<T>(index);
        }
     
        TT *data;
    };

    template <typename T>
    class MemoryPartitioner
    {
        public:
        __device__ MemoryPartitioner(T *inMemory)
        {
            memory = inMemory; 
        }

        __device__ LocalArray<T> array(int size)
        {
            T *arr = &(memory[consumed]);
            consumed += size;
            return {arr};
        }
        private:
        T *memory;
        unsigned int consumed{0};
    };

     template <typename T>
        class PixelArray
        {
            public:
            __device__ PixelArray(){};
            __device__ PixelArray(uchar4 pixel) : channels{T(pixel.x), T(pixel.y), T(pixel.z), T(pixel.w)}{};
            T channels[CHANNEL_COUNT]{0,0,0,0};
            __device__ T& operator[](int index){return channels[index];}
          
             __device__ uchar4 uch4() 
            {
                uchar4 result;
                auto data = reinterpret_cast<unsigned char*>(&result);
                for(int i=0; i<CHANNEL_COUNT; i++)
                    data[i] = __half2int_rn(channels[i]);
                return result;
            }
           
            __device__ void addWeighted(T weight, PixelArray<T> value) 
            {    
                for(int j=0; j<CHANNEL_COUNT; j++)
                    //channels[j] += value[j]*weight;
                    channels[j] = __fmaf_rn(value[j], weight, channels[j]);
            }
            
            __device__ PixelArray<T> operator/= (const T &value)
            {
                for(int j=0; j<CHANNEL_COUNT; j++)
                    this->channels[j] /= value;
                return *this;
            }
            
            __device__ PixelArray<T> operator+= (const PixelArray<T> &value)
            {
                for(int j=0; j<CHANNEL_COUNT; j++)
                    this->channels[j] += value.channels[j];
                return *this;
            }
             
            __device__ PixelArray<T> operator+ (const PixelArray &value)
            {
                for(int j=0; j<CHANNEL_COUNT; j++)
                    this->channels[j] += value.channels[j];
                return *this;
            }
            
            __device__ PixelArray<T> operator-(const PixelArray &value)
            {
                for(int j=0; j<CHANNEL_COUNT; j++)
                    this->channels[j] -= value.channels[j];
                return *this;
            }
            
            __device__ PixelArray<T> operator/(const T &value)
            {
                for(int j=0; j<CHANNEL_COUNT; j++)
                    this->channels[j] /= value;
                return *this;
            }
        };

    class Indexer
    {
        public:
        __device__ int linearIDBase(int id, int size)
        {
            return linearCoord = id*size;
        } 
        
        __device__ int linearID(int id, int size)
        {
            return linearCoord + id*size;
        }
        
        __device__ int linearCoordsBase(int2 coords, int width)
        {
            return linearCoord = coords.y*width + coords.x;
        }

        __device__ int linearCoords(int2 coords, int width)
        {
            return linearCoord + coords.y*width + coords.x;
        }
       
        __device__ int linearCoordsY(int coordY, int width)
        {
            return linearCoord + coordY*width;
        }

        __device__ int getBase()
        {
            return linearCoord;
        }

        private:
        int linearCoord{0};
    };

    template <typename T>
    __device__ static void loadWeightsSync(T *inData, T *data, int size)
    {
        Indexer id;
        id.linearCoordsBase(int2{static_cast<int>(threadIdx.x), static_cast<int>(threadIdx.y)}, blockDim.x);
        int i = id.getBase();
        if(i < size)
        {
            int *intLocal = reinterpret_cast<int*>(data);
            int *intIn = reinterpret_cast<int*>(inData);
            intLocal[i] = intIn[i]; 
        }
        __syncthreads();
    }

    __device__ bool coordsOutside(int2 coords)
    {
        int2 resolution = imgRes();
        return (coords.x >= resolution.x || coords.y >= resolution.y);
    }

    __device__ int2 getImgCoords()
    {
        int2 coords;
        coords.x = (threadIdx.x + blockIdx.x * blockDim.x);
        coords.y = (threadIdx.y + blockIdx.y * blockDim.y);
        return coords;
    }
   
    namespace Pixel
    {
        template <typename T>
        __device__ float distance(PixelArray<T> a, PixelArray<T> b)
        {
            return max(max(abs(a[0]-b[0]), abs(a[1]-b[1])), abs(a[2]-b[2]));
        }

        __device__ void store(uchar4 px, int imageID, int2 coords, cudaSurfaceObject_t *surfaces)
        {
            if constexpr (GUESS_HANDLES)
                surf2Dwrite<uchar4>(px, imageID+1+gridSize(), coords.x*sizeof(uchar4), coords.y);
            else    
                surf2Dwrite<uchar4>(px, surfaces[imageID+gridSize()], coords.x*sizeof(uchar4), coords.y);
        }

        template <typename T>
        __device__ PixelArray<T> load(int imageID, int2 coords, cudaTextureObject_t *surfaces)
        {
            constexpr int MULT_FOUR_SHIFT{2};
            if constexpr (GUESS_HANDLES)
                return PixelArray<T>{surf2Dread<uchar4>(imageID+1, coords.x<<MULT_FOUR_SHIFT, coords.y, cudaBoundaryModeClamp)};
            else    
                return PixelArray<T>{surf2Dread<uchar4>(surfaces[imageID], coords.x<<MULT_FOUR_SHIFT, coords.y, cudaBoundaryModeClamp)};
        }
       
        /* 
        template <typename T>
        __device__ PixelArray<T> loadPx(int imageID, float2 coords, cudaTextureObject_t *textures)
        {
            if constexpr (GUESS_HANDLES)
                return PixelArray<T>{tex2D<uchar4>(imageID+1, coords.x+0.5f, coords.y+0.5f)};
            else    
                return PixelArray<T>{tex2D<uchar4>(textures[imageID], coords.x, coords.y)};
        }
        */
    }

    template <typename T>
    class OnlineVariance
    {
        private:
        float n{0};
        PixelArray<T> m;
        float m2{0};
        
        public:
        __device__ void add(PixelArray<T> val)
        {
           float distance = Pixel::distance(m, val);
           n++;
           PixelArray<T> delta = val-m;
           m += delta/static_cast<T>(n);
           m2 += distance * Pixel::distance(m, val);
        }
        __device__ float variance()
        {
            return m2/(n-1);    
        }      
        __device__ OnlineVariance& operator+=(const PixelArray<T>& rhs){

          add(rhs);
          return *this;
        }
    };

    __device__ int2 focusCoords(int gridID, int2 pxCoords, int focus)
    {
        float2 offset = offsets[gridID];
        int2 coords{static_cast<int>(pxCoords.x-round(offset.x*focus)), static_cast<int>(round(pxCoords.y-offset.y*focus))};
        return coords;
    }

    namespace FocusLevel
    {
        __device__ float evaluatePixel(int2 coords, int focus, cudaSurfaceObject_t *surfaces)
        {
            auto cr = colsRows();
            OnlineVariance<float> variance;
            int gridID = 0; 
            for(int row=0; row<cr.y; row++) 
            {     
                gridID = row*cr.x;
                for(int col=0; col<cr.x; col++) 
                {
                    auto px{Pixel::load<float>(gridID, focusCoords(gridID, coords, focus), surfaces)};
                    variance += px;
                    gridID++;
                }
            }
            return variance.variance();
        }
        
        __device__ float evaluateBlock(int2 coords, int focus, cudaSurfaceObject_t *surfaces)
        {
            constexpr int BLOCK_SIZE{5};
            const int2 BLOCK_OFFSETS[]{ {0,0}, {-1,1}, {1,1}, {-1,-1}, {1,-1}};//, {0,0}, {0,1}, {0,-1}, {-1,0}, {1,0} };
            auto cr = colsRows();
            OnlineVariance<float> variance[BLOCK_SIZE];
            int gridID = 0; 
            for(int row=0; row<cr.y; row++) 
            {     
                gridID = row*cr.x;
                for(int col=0; col<cr.x; col++) 
                {
                    for(int blockPx=0; blockPx<BLOCK_SIZE; blockPx++)
                    {
                        int2 inBlockCoords{coords.x+BLOCK_OFFSETS[blockPx].x, coords.y+BLOCK_OFFSETS[blockPx].y};
                        auto px{Pixel::load<float>(gridID, focusCoords(gridID, inBlockCoords, focus), surfaces)};
                        variance[blockPx] += px;
                    }
                    gridID++;
                }
            }
            
            float finalVariance{0};
            for(int blockPx=0; blockPx<BLOCK_SIZE; blockPx++)
                finalVariance += variance[blockPx].variance();
            return finalVariance;
        }
        
        __device__ float evaluate(int2 coords, int focus, cudaSurfaceObject_t *surfaces, bool blockSampling)
        {
            if(blockSampling)
                return evaluateBlock(coords, focus, surfaces);
            else
                return evaluatePixel(coords, focus, surfaces);
        }
        
        __device__ uchar4 render(int2 coords, int focus, cudaSurfaceObject_t *surfaces, float *weights)
        {
            auto cr = colsRows();
            PixelArray<float> sum;
            int gridID = 0; 
            for(int row=0; row<cr.y; row++) 
            {     
                gridID = row*cr.x;
                for(int col=0; col<cr.x; col++) 
                {
                    auto px{Pixel::load<float>(gridID, focusCoords(gridID, coords, focus), surfaces)};
                    sum.addWeighted(weights[gridID], px);
                    gridID++;
                }
            }
            return sum.uch4();
        }
    }
    
    namespace Focusing
    {
        __device__ int bruteForce(int2 coords, int steps, int range, bool closestViews, bool blockSampling, cudaTextureObject_t *textures, cudaSurfaceObject_t *surfaces, int *closestCoords)
        {
            float stepSize{static_cast<float>(range)/steps};
            float focus{0.0f};
            float minVariance{99999999999999999.0f};
            int optimalFocus{0};
            
            for(int step=0; step<steps; step++)
            {
                int pxFocus = round(focus);
                float variance = FocusLevel::evaluate(coords, pxFocus, surfaces, blockSampling);
                if(variance < minVariance)
                {
                   minVariance = variance;
                   optimalFocus = pxFocus; 
                }
                focus += stepSize;  
            }
            return optimalFocus;
        }
    }

    __global__ void process(FocusMethod method, float methodParameter, bool closestViews, bool blockSampling, cudaTextureObject_t *textures, cudaSurfaceObject_t *surfaces, float *weights, float *closestWeights, int *closestCoords, int range)
    {
        int2 coords = getImgCoords();
        if(coordsOutside(coords))
            return;

        //MemoryPartitioner<half> memoryPartitioner(localMemory);
        //auto localWeights = memoryPartitioner.array(gridSize());
        //loadWeightsSync<half>(weights, localWeights.data, gridSize()/2);

        int focus{0};
        switch(method)
        {
            case ONE_DISTANCE:
                focus = static_cast<int>(methodParameter);
            break;

            case BRUTE_FORCE:
                focus = Focusing::bruteForce(coords, static_cast<int>(methodParameter), range, closestViews, blockSampling, textures, surfaces, closestCoords);
            break;

            default:
            break;
        }        
        uchar4 color = FocusLevel::render(coords, focus, surfaces, weights);
        unsigned char focusColor = (static_cast<float>(focus)/range)*UCHAR_MAX;
        Pixel::store(uchar4{focusColor, focusColor, focusColor, UCHAR_MAX}, focusMapID(), coords, surfaces);
        Pixel::store(color, renderImageID(), coords, surfaces);
    }
}
