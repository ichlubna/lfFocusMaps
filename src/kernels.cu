#include <glm/glm.hpp>
#include <cuda_fp16.h>

namespace Kernels
{
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
   
    template <typename T>
    __device__ PixelArray<T> loadPx(int imageID, int2 coords, cudaTextureObject_t *surfaces)
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
    
    template <typename T>
    __device__ float pixelDistance(PixelArray<T> a, PixelArray<T> b)
    {
        return max(max(abs(a[0]-b[0]), abs(a[1]-b[1])), abs(a[2]-b[2]));
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
           float distance = pixelDistance(m, val);
           n++;
           PixelArray<T> delta = val-m;
           m += delta/static_cast<T>(n);
           m2 += distance * pixelDistance(m, val);
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
        //float2 offset{(currentColRow.x-viewColRow.x)/colsRows().x, (currentColRow.y-viewColRow.y)/colsRows().y};
        float2 offset = offsets[gridID];
        int2 coords{static_cast<int>(round(offset.x*focus))+pxCoords.x, static_cast<int>(round(offset.y*focus))+pxCoords.y};
        return coords;
    }

    __device__ float evaluateFocusLevel(int2 coords, float2 viewColRow, int focus, cudaSurfaceObject_t *surfaces)
    {
        auto cr = colsRows();
        OnlineVariance<float> variance;
        int gridID = 0; 
        for(int row=0; row<cr.y; row++) 
        {     
            gridID = row*cr.x;
            for(int col=0; col<cr.x; col++) 
            {
                auto px{loadPx<float>(gridID, focusCoords(gridID, coords, focus), surfaces)};
                variance += px;
                gridID++;
            }
        }
        return variance.variance();
    }
    
    __device__ uchar4 renderFocusLevel(int2 coords, float2 viewColRow, int focus, cudaSurfaceObject_t *surfaces, float *weights)
    {
        auto cr = colsRows();
        PixelArray<float> sum;
        int gridID = 0; 
        for(int row=0; row<cr.y; row++) 
        {     
            gridID = row*cr.x;
            for(int col=0; col<cr.x; col++) 
            {
                auto px{loadPx<float>(gridID, focusCoords(gridID, coords, focus), surfaces)};
                sum.addWeighted(weights[gridID], px);
                gridID++;
            }
        }
        return sum.uch4();
    }
    
    __device__ void bruteForceScan()
    {

    }

    __device__ void storePx(uchar4 px, int imageID, int2 coords, cudaSurfaceObject_t *surfaces)
    {
        if constexpr (GUESS_HANDLES)
            surf2Dwrite<uchar4>(px, imageID+1+gridSize(), coords.x*sizeof(uchar4), coords.y);
        else    
            surf2Dwrite<uchar4>(px, surfaces[imageID+gridSize()], coords.x*sizeof(uchar4), coords.y);
    }

    __global__ void process(cudaTextureObject_t *textures, cudaSurfaceObject_t *surfaces, float *weights, float *closestWeights, int *closestCoords)
    {
        int2 coords = getImgCoords();
        if(coordsOutside(coords))
            return;

        //MemoryPartitioner<half> memoryPartitioner(localMemory);
        //auto localWeights = memoryPartitioner.array(gridSize());
        //loadWeightsSync<half>(weights, localWeights.data, gridSize()/2);

        uchar4 color = renderFocusLevel(coords, {3.5,3.5}, 0, surfaces, weights);

        storePx(color, focusMapID(), coords, surfaces);
        storePx(color, renderImageID(), coords, surfaces);
    }

}
