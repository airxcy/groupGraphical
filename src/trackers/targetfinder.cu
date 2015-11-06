#include <thrust/device_ptr.h>
#include <thrust/sort.h>
//#include "cuda.h"
//#include "cuda_runtime.h"

#include "opencv2/gpu/device/common.hpp"
#include "opencv2/gpu/device/utility.hpp"
//#include "opencv2/core/cuda_devptrs.hpp"


using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cv::gpu::device;
texture<float, cudaTextureType2D, cudaReadModeElementType> eigTex(0, cudaFilterModePoint, cudaAddressModeClamp);

__device__ uint g_counter = 0;

template <class Mask> __global__ void findCorners(float threshold, const Mask mask, float2* corners, uint max_count, int rows, int cols)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < rows - 1 && j > 0 && j < cols - 1 && mask(i, j))
    {
        float val = tex2D(eigTex, j, i);

        if (val > threshold)
        {
            float maxVal = val;

            maxVal = ::fmax(tex2D(eigTex, j - 1, i - 1), maxVal);
            maxVal = ::fmax(tex2D(eigTex, j    , i - 1), maxVal);
            maxVal = ::fmax(tex2D(eigTex, j + 1, i - 1), maxVal);

            maxVal = ::fmax(tex2D(eigTex, j - 1, i), maxVal);
            maxVal = ::fmax(tex2D(eigTex, j + 1, i), maxVal);

            maxVal = ::fmax(tex2D(eigTex, j - 1, i + 1), maxVal);
            maxVal = ::fmax(tex2D(eigTex, j    , i + 1), maxVal);
            maxVal = ::fmax(tex2D(eigTex, j + 1, i + 1), maxVal);

            if (val == maxVal)
            {
                const uint ind = atomicInc(&g_counter, (uint)(-1));

                if (ind < max_count)
                    corners[ind] = make_float2(j, i);
            }
        }
    }
}

int findCorners_gpu(PtrStepSzf eig, float threshold, PtrStepSzb mask, float2* corners, int max_count)
{
    void* counter_ptr;
    cudaSafeCall( cudaGetSymbolAddress(&counter_ptr, g_counter) );

    cudaSafeCall( cudaMemset(counter_ptr, 0, sizeof(uint)) );

    bindTexture(&eigTex, eig);

    dim3 block(32, 32);
    dim3 grid(divUp(eig.cols, block.x), divUp(eig.rows, block.y));

    if (mask.data)
        findCorners<<<grid, block>>>(threshold, SingleMask(mask), corners, max_count, eig.rows, eig.cols);
    else
        findCorners<<<grid, block>>>(threshold, WithOutMask(), corners, max_count, eig.rows, eig.cols);

    cudaSafeCall( cudaGetLastError() );

    cudaSafeCall( cudaDeviceSynchronize() );

    uint count;
    cudaSafeCall( cudaMemcpy(&count, counter_ptr, sizeof(uint), cudaMemcpyDeviceToHost) );

    return min(count, max_count);
}

class EigGreater
{
public:
    __device__ __forceinline__ bool operator()(float2 a, float2 b) const
    {
        return tex2D(eigTex, a.x, a.y) > tex2D(eigTex, b.x, b.y);
    }
};


void sortCorners_gpu(PtrStepSzf eig, float2* corners, int count)
{
    bindTexture(&eigTex, eig);

    thrust::device_ptr<float2> ptr(corners);

    thrust::sort(ptr, ptr + count, EigGreater());
}
