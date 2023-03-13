#include <glm/glm.hpp>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include "methods.h"

namespace Kernels
{
    constexpr int BLOCK_SAMPLE_COUNT{5};
    constexpr int PIXEL_SAMPLE_COUNT{1};
    constexpr int CLOSEST_COUNT{4}; 

    namespace Constants
    {
        enum TextureMode{NORMAL, MIP, SECONDARY};
        __device__ TextureMode textureMode{NORMAL};
        __device__ void setNormalTextures(){textureMode = NORMAL;};
        __device__ void setMipTextures(){textureMode = MIP;};
        __device__ void setSecondaryTextures(){textureMode = SECONDARY;};

        __constant__ int intConstants[IntConstantIDs::INT_CONSTANTS_COUNT];
        __device__ int2 imgRes(){return {intConstants[IntConstantIDs::IMG_RES_X], intConstants[IntConstantIDs::IMG_RES_Y]};}
        __device__ int2 colsRows(){return{intConstants[IntConstantIDs::COLS], intConstants[IntConstantIDs::ROWS]};}
        __device__ int gridSize(){return intConstants[IntConstantIDs::GRID_SIZE];}
        __device__ int distanceOrder(){return intConstants[IntConstantIDs::DISTANCE_ORDER];}
        __device__ ScanMetric scanMetric(){return static_cast<ScanMetric>(intConstants[IntConstantIDs::SCAN_METRIC]);}
        __device__ FocusMethod focusMethod(){return static_cast<FocusMethod>(intConstants[IntConstantIDs::FOCUS_METHOD]);}
        __device__ bool closestViews(){return intConstants[IntConstantIDs::CLOSEST_VIEWS];}
        __device__ bool blockSampling(){return intConstants[IntConstantIDs::BLOCK_SAMPLING];}
        __device__ ColorDistance YUVDistance(){return static_cast<ColorDistance>(intConstants[IntConstantIDs::YUV_DISTANCE]);}
        __device__ bool blendAddressMode(){return intConstants[IntConstantIDs::BLEND_ADDRESS_MODE];}
        __device__ bool noMap(){return intConstants[IntConstantIDs::NO_MAP];}
        __device__ int ClockSeed(){return intConstants[IntConstantIDs::CLOCK_SEED];}
        
        __constant__ void* dataPointers[DataPointersIDs::POINTERS_COUNT];
        __device__ cudaSurfaceObject_t* surfaces(){return reinterpret_cast<cudaSurfaceObject_t*>(dataPointers[DataPointersIDs::SURFACES]);}
        __device__ cudaTextureObject_t* textures()
        {
            switch(textureMode)
            {
                case NORMAL:
                    return reinterpret_cast<cudaTextureObject_t*>(dataPointers[DataPointersIDs::TEXTURES]);
                break;
                case MIP:
                    return reinterpret_cast<cudaTextureObject_t*>(dataPointers[DataPointersIDs::MIP_TEXTURES]);
                break;
                case SECONDARY:
                    return reinterpret_cast<cudaTextureObject_t*>(dataPointers[DataPointersIDs::SECONDARY_TEXTURES]);
                break;
            }
        }
        __device__ float* closestWeights(){return reinterpret_cast<float*>(dataPointers[DataPointersIDs::CLOSEST_WEIGHTS]);}
        __device__ float* weights(){return reinterpret_cast<float*>(dataPointers[DataPointersIDs::WEIGHTS]);}
        __device__ int* closestCoords(){return reinterpret_cast<int*>(dataPointers[DataPointersIDs::CLOSEST_COORDS]);}
        
        __constant__ float floatConstants[FloatConstantIDs::FLOAT_CONSTANTS_COUNT];
        __device__ float scanSpace(){return floatConstants[FloatConstantIDs::SPACE];}
        __device__ float descentStartStep(){return floatConstants[FloatConstantIDs::DESCENT_START_STEP];}
        __device__ float scanRange(){return floatConstants[FloatConstantIDs::SCAN_RANGE];}
        __device__ float focusMethodParameter(){return floatConstants[FloatConstantIDs::FOCUS_METHOD_PARAMETER];}

        __constant__ float descentStartPoints[DESCENT_START_POINTS];
        __constant__ float hierarchySteps[HIERARCHY_DIVISIONS];
        __constant__ int hierarchySamplings[HIERARCHY_DIVISIONS];
    }

    __device__ constexpr int MAX_IMAGES{256};
    __constant__ float2 offsets[MAX_IMAGES];
    //extern __shared__ half localMemory[];

     template <typename T>
        class PixelArray
        {
            public:
            __device__ PixelArray(){};
            __device__ PixelArray(uchar4 pixel) : channels{T(pixel.x), T(pixel.y), T(pixel.z), T(pixel.w)}{};
            __device__ PixelArray(float4 pixel) : channels{T(pixel.x), T(pixel.y), T(pixel.z), T(pixel.w)}{};
            T channels[CHANNEL_COUNT]{0,0,0,0};
            __device__ T& operator[](int index){return channels[index];}
          
             __device__ uchar4 uch4() 
            {
                uchar4 result;
                auto data = reinterpret_cast<unsigned char*>(&result);
                for(int i=0; i<CHANNEL_COUNT; i++)
                    data[i] = __float2int_rn(channels[i]);
                return result;
            }
           
            __device__ void addWeighted(T weight, PixelArray<T> &value) 
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
            
            __device__ PixelArray<T> operator*(const T &value)
            {
                for(int j=0; j<CHANNEL_COUNT; j++)
                    this->channels[j] *= value;
                return *this;
            }
        };

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
        template <typename T>        
        __device__ uchar4 RGBtoYUV(PixelArray<T> &rgb)
        {
            uchar4 yuv;
            yuv.x = ( __float2uint_rn(  66 * rgb[0] + 129 * rgb[1] +  25 * rgb[2] + 128) >> 8) +  16;
            yuv.y = ( __float2uint_rn( -38 * rgb[0] -  74 * rgb[1] + 112 * rgb[2] + 128) >> 8) + 128;
            yuv.z = ( __float2uint_rn( 112 * rgb[0] -  94 * rgb[1] -  18 * rgb[2] + 128) >> 8) + 128;
            return yuv;
        }
        
        template <typename T>        
        __device__ unsigned char RGBtoY(PixelArray<T> &rgb)
        {
            return (__float2uint_rn(  66 * rgb[0] + 129 * rgb[1] +  25 * rgb[2] + 128) >> 8) +  16;
        }


        template <typename T>
        __device__ float distance(PixelArray<T> &a, PixelArray<T> &b)
        {
            float dist{0};
            switch(Constants::YUVDistance())
            {
                case RGB:
                {
                    dist = max(max(abs(a[0]-b[0]), abs(a[1]-b[1])), abs(a[2]-b[2]));
                }
                break;
                
                case YUV:            
                {
                    auto yuvA = RGBtoYUV<T>(a);
                    auto yuvB = RGBtoYUV<T>(b);
                    dist = max(max(abs(yuvA.x-yuvB.x), abs(yuvA.y-yuvB.y)), abs(yuvA.z-yuvB.z));
                }
                break;
                
                case YUVw:            
                {
                    auto yuvA = RGBtoYUV(a);
                    auto yuvB = RGBtoYUV(b);
                    dist = max(max(abs(yuvA.x-yuvB.x)>>2, abs(yuvA.y-yuvB.y)), abs(yuvA.z-yuvB.z));
                }
                break;

                case Y:
                    dist = abs(RGBtoY<T>(a)-RGBtoY<T>(b));
                break;
            }
            return __powf (dist, Constants::distanceOrder());
        }

        __device__ void store(uchar4 px, int imageID, int2 coords)
        {
            surf2Dwrite<uchar4>(px, Constants::surfaces()[imageID], coords.x*sizeof(uchar4), coords.y);
        }

        template <typename T>
        __device__ PixelArray<T> load(int imageID, float2 coords)
        {
            int id = Constants::textures()[imageID];
            if(Constants::blendAddressMode())
            {
                constexpr float SPREAD{0.0015f};
                float offset{0};
                if(coords.x > 1.0 || coords.x < 0.0)
                    offset += floor(coords.x)*SPREAD;
                if(coords.y > 1.0 || coords.y < 0.0)
                    offset += floor(coords.y)*SPREAD;
                PixelArray<T> pixel;
                const float2 offsets[4] = {{offset, offset}, {offset, -offset}, {-offset, -offset}, {-offset, offset}};
                for(int i=0; i<4; i++)
                    pixel += PixelArray<T>{tex2D<float4>(id, coords.x+offsets[i].x, coords.y+offsets[i].y)}; 
                return (pixel/4)*UCHAR_MAX;
            }    
            else
                return PixelArray<T>{tex2D<float4>(id, coords.x, coords.y)}*UCHAR_MAX;
        } 
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
            __device__ float dispersionAmount()
            {
                return m2/(n-1);    
            }      
            __device__ OnlineVariance& operator+=(const PixelArray<T>& rhs){

              add(rhs);
              return *this;
            }
        };
        
        template <typename T>
        class Range
        {
            private:
            PixelArray<T> minCol{float4{FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX}};
            PixelArray<T> maxCol{float4{FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN}};
            
            public:
            __device__ void add(PixelArray<T> val)
            {
                minCol[0] = min(minCol[0],val[0]);
                minCol[1] = min(minCol[1],val[1]);
                minCol[2] = min(minCol[2],val[2]);
                maxCol[0] = max(maxCol[0],val[0]);
                maxCol[1] = max(maxCol[1],val[1]);
                maxCol[2] = max(maxCol[2],val[2]);
            }
            __device__ float dispersionAmount()
            {
                return Pixel::distance(minCol, maxCol);  
            }      
            __device__ Range& operator+=(const PixelArray<T>& rhs){

              add(rhs);
              return *this;
            }
        };
       
        template <typename T>
        class Percentile
        {
            private:
            float stepUp;
            float stepDown;
            float step{10};
            float value{0};
            bool init{false};

            public:
            __device__ Percentile(float percentile)
            {
                stepUp = 1.0f-percentile;
                stepDown = percentile;
            }
            __device__ void add(PixelArray<T> pixel)
            {
                PixelArray<T> origin;
                float dist = Pixel::distance<T>(pixel, origin);

                if(!init)
                {
                    value = dist;
                    step = max(dist, 1.0f);
                    init = true;
                    return;
                }
                if(value > dist)
                    value -= step*stepUp;
                else if(value < dist)
                    value += step*stepDown;
                if(abs(value-dist) < step)
                   step /= 2.0f; 
            }
            __device__ float result()
            {
                return value;
            }
        };
 
        template <typename T>
        class IQR
        {
            private:
            Percentile<T> first{0.25};            
            Percentile<T> second{0.75};            
 
            public:
            __device__ void add(PixelArray<T> val)
            {
                first.add(val); 
                second.add(val); 
            }
            __device__ float dispersionAmount()
            {
                return second.result()-first.result();
            }      
            __device__ IQR& operator+=(const PixelArray<T>& rhs){

              add(rhs);
              return *this;
            }
        };
      
        template <typename T>
        class Mad
        {
            private:
            PixelArray<T> last;
            PixelArray<T> sample;
            float dist{0};
            int n{0};
            static constexpr int SAMPLE_CYCLE{5};
             
            public:
            __device__ void add(PixelArray<T> val)
            { 
                if(n%SAMPLE_CYCLE == 0)
                    sample = val;
                if(n != 0)
                {
                    dist += Pixel::distance(val, sample);
                    dist += Pixel::distance(val, last); 
                }
                last = val; 
                n++;
            }
            __device__ float dispersionAmount()
            {
                return dist;
            }      
            __device__ Mad& operator+=(const PixelArray<T>& rhs){

              add(rhs);
              return *this;
            }
        };
    }

    template<typename T>
    __device__ float2 focusCoords(int gridID, T pxCoords, float focus)
    {
        float2 offset = offsets[gridID];
        float2 coords{pxCoords.x-offset.x*focus, pxCoords.y-offset.y*focus};
        return coords;
    }

    __device__ float transformFocus(float focus, int range, float space)
    {
        if(space != 1.0f)
        {
            float normalized = focus/range;
            return __powf(normalized, space)*range;
        }
        return focus;
    }

    namespace FocusLevel
    {      
        template<int blockSize, typename T> 
        __device__ void evaluateBlock(int gridID, float focus, float2 coords, T *dispersions)
        {
            float transformedFocus = transformFocus(focus, Constants::scanRange(), Constants::scanSpace());
            const float2 BLOCK_OFFSETS[]{ {0,0}, {-1,0.5}, {0.5, 1}, {1,-0.5}, {-0.5,-1} };
            for(int blockPx=0; blockPx<blockSize; blockPx++)
            {
                float2 inBlockCoords{coords.x+BLOCK_OFFSETS[blockPx].x, coords.y+BLOCK_OFFSETS[blockPx].y};
                auto px{Pixel::load<float>(gridID, focusCoords(gridID, inBlockCoords, transformedFocus))};
                dispersions[blockPx] += px;
            }
        }

        template<typename T, int blockSize, bool closest=false>
        __device__ float evaluateDispersion(float2 coords, float focus)
        {
            auto cr = Constants::colsRows();
            T dispersionCalc[blockSize];
                
            int gridID = 0;

            if constexpr (closest)
            {  
                auto closestCoords = Constants::closestCoords();
                for(int i=0; i<CLOSEST_COUNT; i++) 
                {     
                    int gridID = closestCoords[i];
                    evaluateBlock<blockSize>(gridID, focus, coords, dispersionCalc);
                }           
            }
            else
                for(int row=0; row<cr.y; row++) 
                {     
                    gridID = row*cr.x;
                    for(int col=0; col<cr.x; col++) 
                    {
                        evaluateBlock<blockSize>(gridID, focus, coords, dispersionCalc);
                        gridID++;
                    }
                }

            float finalDispersion{0};
            for(int blockPx=0; blockPx<blockSize; blockPx++)
                finalDispersion += dispersionCalc[blockPx].dispersionAmount();
            return finalDispersion;
        }

        template<int blockSize, bool closest, typename...TAIL>
        __device__ typename std::enable_if_t<sizeof...(TAIL)==0, float> 
        call(int,ScanMetric,float2,float){}

        template<int blockSize, bool closest, typename H, typename...TAIL>
        __device__ float call(int n,ScanMetric type, float2 coords, float focus)
        {
            if(n==type)
                return evaluateDispersion<H, blockSize, closest>(coords, focus);
            return call<blockSize, closest, TAIL...>(n+1,type, coords, focus);
        }

        template<int blockSize, bool closest=false>
        __device__ float dispersion(ScanMetric t, float2 coords, float focus)
        {
            return call<blockSize, closest, ScanMetrics::OnlineVariance<float>, ScanMetrics::Range<float>, ScanMetrics::IQR<float>, ScanMetrics::Mad<float>>(0,t, coords, focus);
        }

        __device__ float evaluate(float2 coords, float focus)
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
        __device__ uchar4 render(float2 coords, float focus)
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
        class Optimum
        {
            public:
            float optimalFocus{0};
            float minDispersion{FLT_MAX};
            __device__ bool add(float focus, float dispersion)
            {
                if(dispersion < minDispersion)
                {
                   minDispersion = dispersion;
                   optimalFocus = focus; 
                   return true;
                }
                return false;
            }
            __device__ void addForce(float focus, float dispersion)
            {
                   minDispersion = dispersion;
                   optimalFocus = focus; 
            }
        }; 

        __device__ Optimum& minOpt(Optimum &a, Optimum &b)
        {
            if(a.minDispersion < b.minDispersion)
                return a;
            else
                return b;
        }

        __device__ float bruteForce(float2 coords)
        {
            int steps = Constants::focusMethodParameter();
            float stepSize{static_cast<float>(Constants::scanRange())/steps};
            float focus{0.0f};
            Optimum optimum;
            
            for(int step=0; step<steps; step++)
            {
                float dispersion = FocusLevel::evaluate(coords, focus);
                optimum.add(focus, dispersion);
                focus += stepSize;  
            }
            return optimum.optimalFocus;
        }
        
        __device__ float randomSampling(float2 coords)
        {
            unsigned int linearID = coords.y*Constants::imgRes().x + coords.x;
            curandState state;
            curand_init(Constants::ClockSeed()+linearID, 0, 0, &state);
            int steps = Constants::focusMethodParameter();
            int range = Constants::scanRange();
            Optimum optimum;
            
            for(int step=0; step<steps; step++)
            {
                float focus = range*curand_uniform(&state) ;
                float dispersion = FocusLevel::evaluate(coords, focus);
                optimum.add(focus, dispersion);
            }

            return optimum.optimalFocus;
        }
       
        template <bool stochastic> 
        __device__ float hierarchical(float2 coords)
        {
            curandState state;
            if constexpr (stochastic)
            {
                unsigned int linearID = coords.y*Constants::imgRes().x + coords.x;
                curand_init(Constants::ClockSeed()+linearID, 0, 0, &state);
            }

            int range = Constants::scanRange();
            Optimum optimum; 
            bool divide{true};
            float2 dividedRange{0, static_cast<float>(range)};
            for(int d=0; d<HIERARCHY_DIVISIONS; d++)
            {  
                Optimum leftRightOptimum[2];
                int sampling = Constants::hierarchySamplings[d];
                int samplingHalf = sampling/2;
                float focus = dividedRange.x;
                for(int i=0; i<sampling; i++)
                {
                    float disp = FocusLevel::evaluate(coords, focus);
                    leftRightOptimum[(i<samplingHalf) ? 0 : 1].add(focus, disp);
                    focus += Constants::hierarchySteps[d];
                }

                if(leftRightOptimum[0].minDispersion < leftRightOptimum[1].minDispersion)
                {
                    divide = optimum.add(leftRightOptimum[0].optimalFocus, leftRightOptimum[0].minDispersion);
                    dividedRange = {0, dividedRange.y*0.5f};
                }
                else
                {
                    divide = optimum.add(leftRightOptimum[1].optimalFocus, leftRightOptimum[1].minDispersion);
                    dividedRange = {dividedRange.y*0.5f, dividedRange.y};
                }
                if(!divide)
                    break;
            }
            return optimum.optimalFocus;
        }
        
        template <bool stochastic> 
        __device__ float descent(float2 coords)
        {
            curandState state;
            if constexpr (stochastic)
            {
                unsigned int linearID = coords.y*Constants::imgRes().x + coords.x;
                curand_init(Constants::ClockSeed()+linearID, 0, 0, &state);
            }

            constexpr int MAX_STEPS{100};
            Optimum optimum[DESCENT_START_POINTS];
            constexpr float LEARN_RATE{0.1f};
            constexpr float MIN_STEP{0.5f};
            
            for(int p=0; p<DESCENT_START_POINTS; p++)
            {
                float focus = Constants::descentStartPoints[p];
                float step{Constants::descentStartStep()};
                for(int i=0; i<MAX_STEPS; i++)
                {      
                    float2 focusPair{focus-step, focus+step};
                    float2 dispersionPair{FocusLevel::evaluate(coords, focusPair.x), FocusLevel::evaluate(coords, focusPair.y)};
                    if(dispersionPair.x < dispersionPair.y)
                        optimum[p].addForce(focusPair.x, dispersionPair.x);
                    else
                        optimum[p].addForce(focusPair.y, dispersionPair.y);
                    step = LEARN_RATE*abs(focus-optimum[p].optimalFocus);
                    focus = optimum[p].optimalFocus;
                    if(step < MIN_STEP)
                        break;
                }
            }
            Optimum &minimal = optimum[0];
            for(int i=1; i<DESCENT_START_POINTS; i++)
                minimal = minOpt(minimal, optimum[i]);
            return minimal.optimalFocus;
        }
        
        template <bool stochastic> 
        __device__ float pyramid(float2 coords)
        {
            curandState state;
            if constexpr (stochastic)
            {
                unsigned int linearID = coords.y*Constants::imgRes().x + coords.x;
                curand_init(Constants::ClockSeed()+linearID, 0, 0, &state);
            }
        }
    }

    __global__ void process()
    {
        int2 threadCoords = getImgCoords(); 
        if(coordsOutside(threadCoords))
            return;
        float2 coords = {static_cast<float>(threadCoords.x)/Constants::imgRes().x,
                        static_cast<float>(threadCoords.y)/Constants::imgRes().y};

        //MemoryPartitioner<half> memoryPartitioner(localMemory);
        //auto localWeights = memoryPartitioner.array(gridSize());
        //loadWeightsSync<half>(weights, localWeights.data, gridSize()/2);

        float focus{0};
        Constants::setSecondaryTextures();
        switch(Constants::focusMethod())
        {
            case ONE_DISTANCE:
                focus = Constants::focusMethodParameter();
            break;

            case BRUTE_FORCE:
                focus = Focusing::bruteForce(coords);
            break;
            
            case RANDOM:
                focus = Focusing::randomSampling(coords);
            break;
            
            case PYRAMID:
                if(Constants::focusMethodParameter())
                    focus = Focusing::pyramid<true>(coords);
                else
                    focus = Focusing::pyramid<false>(coords);
            break;
            
            case HIERARCHY:
                if(Constants::focusMethodParameter())
                    focus = Focusing::hierarchical<true>(coords);
                else
                    focus = Focusing::hierarchical<false>(coords);
            break;
            
            case DESCENT:
                if(Constants::focusMethodParameter())
                    focus = Focusing::descent<true>(coords);
                else
                    focus = Focusing::descent<false>(coords);
            break;

            default:
            break;
        }       

        Constants::setNormalTextures(); 
        uchar4 color{0};
        if(Constants::closestViews())
            color = FocusLevel::render<true>(coords, focus);
        else
            color = FocusLevel::render(coords, focus);

        if(!Constants::noMap())
        {
            unsigned char focusColor = (focus/Constants::scanRange())*UCHAR_MAX;
            Pixel::store(uchar4{focusColor, focusColor, focusColor, UCHAR_MAX}, FileNames::FOCUS_MAP, threadCoords);
        }
        Pixel::store(color, FileNames::RENDER_IMAGE, threadCoords);
    }
}
