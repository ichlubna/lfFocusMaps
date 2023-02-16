#include <glm/glm.hpp>
#include <cuda_fp16.h>
#include "methods.h"

namespace Kernels
{
    constexpr int BLOCK_SAMPLE_COUNT{5};
    constexpr int PIXEL_SAMPLE_COUNT{1};
    constexpr int CLOSEST_COUNT{4}; 
    __device__ constexpr bool GUESS_HANDLES{false};

    namespace Constants
    {
        __constant__ int intConstants[IntConstantIDs::INT_CONSTANTS_COUNT];
        __device__ int2 imgRes(){return {intConstants[IntConstantIDs::IMG_RES_X], intConstants[IntConstantIDs::IMG_RES_Y]};}
        __device__ int2 colsRows(){return{intConstants[IntConstantIDs::COLS], intConstants[IntConstantIDs::ROWS]};}
        __device__ int gridSize(){return intConstants[IntConstantIDs::GRID_SIZE];}
        __device__ int distanceOrder(){return intConstants[IntConstantIDs::DISTANCE_ORDER];}
        __device__ ScanMetric scanMetric(){return static_cast<ScanMetric>(intConstants[IntConstantIDs::SCAN_METRIC]);}
        __device__ FocusMethod focusMethod(){return static_cast<FocusMethod>(intConstants[IntConstantIDs::FOCUS_METHOD]);}
        __device__ int focusMethodParameter(){return intConstants[IntConstantIDs::FOCUS_METHOD_PARAMETER];}
        __device__ int scanRange(){return intConstants[IntConstantIDs::RANGE];}
        __device__ bool closestViews(){return intConstants[IntConstantIDs::CLOSEST_VIEWS];}
        __device__ bool blockSampling(){return intConstants[IntConstantIDs::BLOCK_SAMPLING];}
        __device__ bool YUVDistance(){return intConstants[IntConstantIDs::YUV_DISTANCE];}
        
        __constant__ void* dataPointers[DataPointersIDs::POINTERS_COUNT];
        __device__ cudaSurfaceObject_t* surfaces(){return reinterpret_cast<cudaSurfaceObject_t*>(dataPointers[DataPointersIDs::SURFACES]);}
        __device__ cudaTextureObject_t* textures(){return reinterpret_cast<cudaTextureObject_t*>(dataPointers[DataPointersIDs::TEXTURES]);}
        __device__ float* closestWeights(){return reinterpret_cast<float*>(dataPointers[DataPointersIDs::CLOSEST_WEIGHTS]);}
        __device__ float* weights(){return reinterpret_cast<float*>(dataPointers[DataPointersIDs::WEIGHTS]);}
        __device__ int* closestCoords(){return reinterpret_cast<int*>(dataPointers[DataPointersIDs::CLOSEST_COORDS]);}
        
        __constant__ float floatConstants[FloatConstantIDs::FLOAT_CONSTANTS_COUNT];
        __device__ float scanSpace(){return floatConstants[FloatConstantIDs::SPACE];}
    }

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
        int2 resolution = Constants::imgRes();
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
        //source: https://learn.microsoft.com/en-us/windows/win32/medfound/recommended-8-bit-yuv-formats-for-video-rendering
        __device__ uchar4 RGBtoYUV(uchar4 rgb)
        {
            uchar4 yuv;
            yuv.x = ( (  66 * rgb.x + 129 * rgb.y +  25 * rgb.z + 128) >> 8) +  16;
            yuv.y = ( ( -38 * rgb.x -  74 * rgb.y + 112 * rgb.z + 128) >> 8) + 128;
            yuv.z = ( ( 112 * rgb.x -  94 * rgb.y -  18 * rgb.z + 128) >> 8) + 128;
            return yuv;
        }

        template <typename T>
        __device__ float distance(PixelArray<T> a, PixelArray<T> b)
        {
            float dist{0};
            if(Constants::YUVDistance())
            {
                auto yuvA = RGBtoYUV(a.uch4());
                auto yuvB = RGBtoYUV(b.uch4());
                dist = max(max(abs(yuvA.x-yuvB.x)>>2, abs(yuvA.y-yuvB.y)), abs(yuvA.z-yuvB.z));
            }
            else
                dist = max(max(abs(a[0]-b[0]), abs(a[1]-b[1])), abs(a[2]-b[2]));
            return __powf (dist, Constants::distanceOrder());
        }

        __device__ void store(uchar4 px, int imageID, int2 coords)
        {
            if constexpr (GUESS_HANDLES)
                surf2Dwrite<uchar4>(px, imageID+1+Constants::gridSize(), coords.x*sizeof(uchar4), coords.y);
            else    
                surf2Dwrite<uchar4>(px, Constants::surfaces()[imageID+Constants::gridSize()], coords.x*sizeof(uchar4), coords.y);
        }

        template <typename T>
        __device__ PixelArray<T> load(int imageID, int2 coords)
        {
            constexpr int MULT_FOUR_SHIFT{2};
            if constexpr (GUESS_HANDLES)
                return PixelArray<T>{surf2Dread<uchar4>(imageID+1, coords.x<<MULT_FOUR_SHIFT, coords.y, cudaBoundaryModeClamp)};
            else    
                return PixelArray<T>{surf2Dread<uchar4>(Constants::surfaces()[imageID], coords.x<<MULT_FOUR_SHIFT, coords.y, cudaBoundaryModeClamp)};
        }
       
        /* 
        template <typename T>
        __device__ PixelArray<T> loadPx(int imageID, float2 coords)
        {
            if constexpr (GUESS_HANDLES)
                return PixelArray<T>{tex2D<uchar4>(imageID+1, coords.x+0.5f, coords.y+0.5f)};
            else    
                return PixelArray<T>{tex2D<uchar4>(textures()[imageID], coords.x, coords.y)};
        }
        */
    }

    namespace ScanMetrics
    {
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
               float distance = Pixel::distance<T>(m, val);
               n++;
               PixelArray<T> delta = val-m;
               m += delta/static_cast<T>(n);
               //m2 += distance * Pixel::distance(m, val);
               m2 = __fmaf_rn(distance, Pixel::distance(m, val), m2);

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
    }

    __device__ int2 focusCoords(int gridID, int2 pxCoords, int focus)
    {
        float2 offset = offsets[gridID];
        int2 coords{static_cast<int>(pxCoords.x-round(offset.x*focus)), static_cast<int>(round(pxCoords.y-offset.y*focus))};
        return coords;
    }

    __device__ int transformFocus(int focus, int range, float space)
    {
        if(space != 1.0f)
        {
            float normalized = static_cast<float>(focus)/range;
            return round(__powf(normalized, space)*range);
        }
        return focus;
    }

    namespace FocusLevel
    {      
        template<int blockSize> 
        __device__ void evaluateBlock(int gridID, int focus, int2 coords, ScanMetrics::OnlineVariance<float> *variances)
        {
            int transformedFocus = transformFocus(focus, Constants::scanRange(), Constants::scanSpace());
            const int2 BLOCK_OFFSETS[]{ {0,0}, {-1,1}, {1,1}, {-1,-1}, {1,-1} };//, {0,0}, {0,1}, {0,-1}, {-1,0}, {1,0} };
            for(int blockPx=0; blockPx<blockSize; blockPx++)
            {
                int2 inBlockCoords{coords.x+BLOCK_OFFSETS[blockPx].x, coords.y+BLOCK_OFFSETS[blockPx].y};
                auto px{Pixel::load<float>(gridID, focusCoords(gridID, inBlockCoords, transformedFocus))};
                variances[blockPx] += px;
            }
        }

        template<typename T, int blockSize, bool closest=false>
        __device__ float evaluateDispersion(int2 coords, int focus)
        {
            auto cr = Constants::colsRows();
            T variance[blockSize];
                
            int gridID = 0;

            if constexpr (closest)
            {  
                auto closestCoords = Constants::closestCoords();
                for(int i=0; i<CLOSEST_COUNT; i++) 
                {     
                    int gridID = closestCoords[i];
                    evaluateBlock<blockSize>(gridID, focus, coords, variance);
                }           
            }
            else
                for(int row=0; row<cr.y; row++) 
                {     
                    gridID = row*cr.x;
                    for(int col=0; col<cr.x; col++) 
                    {
                        evaluateBlock<blockSize>(gridID, focus, coords, variance);
                        gridID++;
                    }
                }

            float finalVariance{0};
            for(int blockPx=0; blockPx<blockSize; blockPx++)
                finalVariance += variance[blockPx].variance();
            return finalVariance;
        }

        template<typename...TAIL>
        typename std::enable_if_t<sizeof...(TAIL)==0,void> 
        __device__ call(int,ScanMetric,int2,int){}

        template<typename H, int blockSize, bool closest, typename...TAIL>
        __device__ float call(int n,ScanMetric type, int2 coords, int focus)
        {
            if(n==type)
                return evaluateDispersion<H, blockSize, closest>(coords, focus);
            call<TAIL...>(n+1,type, coords, focus);
        }

        template<int blockSize, bool closest=false>
        __device__ float dispersion(ScanMetric t, int2 coords, int focus)
        {
            return call<ScanMetrics::OnlineVariance<float>, blockSize, closest>(0,t, coords, focus);
        }

        __device__ float evaluate(int2 coords, int focus)
        {
            auto closestViews = Constants::closestViews(); 
            auto blockSampling = Constants::blockSampling();
            auto scanMetric = Constants::scanMetric();
 
            if(closestViews)
                if(blockSampling)
                    return dispersion<BLOCK_SAMPLE_COUNT, true>(scanMetric, coords, focus);
                else
                    return dispersion<PIXEL_SAMPLE_COUNT, true>(scanMetric, coords, focus);
            else
                if(blockSampling)
                    return dispersion<BLOCK_SAMPLE_COUNT>(scanMetric, coords, focus);
                else
                    return dispersion<PIXEL_SAMPLE_COUNT>(scanMetric, coords, focus);
        }
       
        template<bool closest=false>
        __device__ uchar4 render(int2 coords, int focus)
        {
            auto cr = Constants::colsRows();
            PixelArray<float> sum;
            int gridID = 0; 
            
            if constexpr (closest)
            {
                auto closestCoords = Constants::closestCoords();
                auto weights = Constants::closestWeights();
                for(int i=0; i<CLOSEST_COUNT; i++) 
                {
                    gridID = closestCoords[i];
                    auto px{Pixel::load<float>(gridID, focusCoords(gridID, coords, focus))};
                    sum.addWeighted(weights[i], px);
                }
            }
            else
            {
                auto weights = Constants::weights();
                for(int row=0; row<cr.y; row++) 
                {     
                    gridID = row*cr.x;
                    for(int col=0; col<cr.x; col++) 
                    {
                        auto px{Pixel::load<float>(gridID, focusCoords(gridID, coords, focus))};
                        sum.addWeighted(weights[gridID], px);
                        gridID++;
                    }
                }
            }
            return sum.uch4();
        }      
    }
    
    namespace Focusing
    {
        __device__ int bruteForce(int2 coords)
        {
            int steps = Constants::focusMethodParameter();
            float stepSize{static_cast<float>(Constants::scanRange())/steps};
            float focus{0.0f};
            float minVariance{FLT_MAX};
            int optimalFocus{0};
            
            for(int step=0; step<steps; step++)
            {
                int pxFocus = round(focus);
                float variance = FocusLevel::evaluate(coords, pxFocus);
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

    __global__ void process()
    {
        int2 coords = getImgCoords();
        if(coordsOutside(coords))
            return;

        //MemoryPartitioner<half> memoryPartitioner(localMemory);
        //auto localWeights = memoryPartitioner.array(gridSize());
        //loadWeightsSync<half>(weights, localWeights.data, gridSize()/2);

        int focus{0};
        switch(Constants::focusMethod())
        {
            case ONE_DISTANCE:
                focus = Constants::focusMethodParameter();
            break;

            case BRUTE_FORCE:
                focus = Focusing::bruteForce(coords);
            break;

            default:
            break;
        }        
        uchar4 color{0};
        if(Constants::closestViews())
            color = FocusLevel::render<true>(coords, focus);
        else
            color = FocusLevel::render(coords, focus);

        unsigned char focusColor = (static_cast<float>(focus)/Constants::scanRange())*UCHAR_MAX;
        Pixel::store(uchar4{focusColor, focusColor, focusColor, UCHAR_MAX}, FileNames::FOCUS_MAP, coords);
        Pixel::store(color, FileNames::RENDER_IMAGE, coords);
    }
}
