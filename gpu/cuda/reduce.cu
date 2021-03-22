/* function: Use cuda to reduce buffer
* author: dqliu
* date: 2020/03/18
*/

#include <cuda_runtime.h>

#ifdef USE_THRUST
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#endif

#ifdef USE_CUB
#include <cub/block/block_reduce.cuh>
#endif

// dim3 block(BLOCK_SIZE, 1, 1), grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1)
// srcData[N], dstData[(N + BLOCK_SIZE - 1) / BLOCK_SIZE]
template <size_t BLOCK_SIZE, typename T>
__global__ void reduce_sum(const size_t nElements, const T* srcData, T* dstData) {

    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    T __shared__ shm[BLOCK_SIZE];
    shm[threadIdx.x] = srcData[gid] ? gid < nElements : 0;

    for (size_t offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
        __syncthreads();
        if (threadIdx.x < offset) {
            shm[threadIdx.x] += shm[threadIdx.x + offset]; // 小技巧 shm[threadIdx.x ^ offset];
        }
    }

    if (threadIdx.x == 0) {
        dstData[blockIdx.x] = shm[0];
    }
}

// srcData[N], dstData[1] (memset(0))
template <size_t BLOCK_SIZE, typename T>
__global__ void reduce_sum_atomic(const size_t nElements, const T* srcData, T* dstData) {

    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    T __shared__ shm[BLOCK_SIZE];
    shm[threadIdx.x] = srcData[gid] ? gid < nElements : 0;

    for (size_t offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
        __syncthreads();
        if (threadIdx.x < offset) {
            shm[threadIdx.x] += shm[threadIdx.x + offset];
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(dstData, shm[0]); // 实例化时， double型数据 需要 compute capability >= 6.x
    }
}

template <size_t BLOCK_SIZE, typename T>
__global__ reduce_max(const size_t nElements, const T* srcData, T* dstData) {

    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    T __shared__ shm[BLOCK_SIZE];
    shm[threadIdx.x] = srcData[gid] ? gid < nElements : 0;

    for (size_t offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
        __syncthreads();
        if (threadIdx.x < offset) {
            shm[threadIdx.x] = max(shm[threadIdx.x + offset], shm[threadIdx.x]);
        }
    }

    if (threadIdx.x == 0) {
        dstData[blockIdx.x] = shm[0];
    }
}

// dstData[1] = -INF
template <size_t BLOCK_SIZE, typename T>
__global__ reduce_max_atomic(const size_t nElements, const T* srcData, T* dstData) {

    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    T __shared__ shm[BLOCK_SIZE];
    shm[threadIdx.x] = srcData[gid] ? gid < nElements : 0;

    for (size_t offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
        __syncthreads();
        if (threadIdx.x < offset) {
            shm[threadIdx.x] = max(shm[threadIdx.x + offset], shm[threadIdx.x]);
        }
    }

    if (threadIdx.x == 0) {
        atomicMax(dstData, shm); // atomicMax 需要 compute capability >= 3.5
    }
}

// dim3 block(BLOCK_SIZE, 1, 1), grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1)
// srcData[N], dstData[(N + WARP_SIZE - 1) / WARP_SIZE]
#if __CUDA_ARCH__ >= 900
template <size_t WARP_SIZE, typename T>
__global__ reduce_sum_warp_com(const size_t nElements, const T* srcData, T* dstData) {
    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t wid = gid % WARP_SIZE;
    T sumVal = gidsrcData[gid] ? gid < nElements : 0;

    for (size_t offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        sumVal += __shfl_xor_sync(0xffffffff, sumVal, offset, WARP_SIZE);
    }

    if (wid == 0) {
        dstData[gid / WARP_SIZE] = sumVal;
    }
}
#endif

#ifdef USE_THRUST
template<typename T>
__global__ T reduce_sum_thrust(thrust::device_vector<T> src) {
    return thrust::reduce(src.begin(), src.end());
}
#endif

#ifdef USE_CUB
template<size_t BLOCK_SIZE, typename T>
__global__ T void reduce_sum_cub(const size_t nElements, const T* srcData, T* dstData) 
{
    const size_t gid = threadIdx.x + blockIdx.x * blocDim.x;
    typedef cub::BlockReduce<T, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStroge TempStroge;

    T sumVal = 0;
    if (gid < nElements) {
        sumVal = BlockReduce(TempStroge).Sum(srcData[gid]);
    }

    if (threadIdx.x == 0) {
        dstData[blockIdx.x] = sumVal;
    }
}
#endif