/* function: Use cuda to reduce buffer
* author: dqliu
* date: 2020/03/18
*/
#include "cuda_op.h"

// dim3 block(BLOCK_SIZE, 1, 1), grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1)
// srcData[N], dstData[(N + BLOCK_SIZE - 1) / BLOCK_SIZE]
template <size_t BLOCK_SIZE, typename T>
__global__ void reduce_sum(const size_t nElements, const T* srcData, T* dstData) 
{
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

template <>
void cudaCallReduceSUMSharedMem<unsigned int>(const size_t nElements, const unsigned int* srcData, unsigned int* dstData)
{
	const size_t BLOCK_SIZE = 1024;
	reduce_sum<BLOCK_SIZE, unsigned int><<<
				(nElements + BLOCK_SIZE - 1) / BLOCK_SIZE,
				BLOCK_SIZE>>>(
				nElements,
				srcData,
				dstData);
}

// srcData[N], dstData[1] (memset(0))
template <size_t BLOCK_SIZE, typename T>
__global__ void reduce_sum_atomic(const size_t nElements, const T* srcData, T* dstData) 
{
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
__global__ void reduce_max(const size_t nElements, const T* srcData, T* dstData) 
{
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
__global__ void reduce_max_atomic(const size_t nElements, const T* srcData, T* dstData) 
{
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
// #if __CUDA_ARCH__ >= 900
template<size_t WARP_SIZE, typename T>
__global__ void reduce_sum_warp_com(const size_t nElements, const T* srcData, T* dstData)
{
    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t wid = gid % WARP_SIZE;
    T sumVal = gid < nElements ? srcData[gid] : 0;

    for (size_t offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        sumVal += __shfl_xor_sync(0xffffffff, sumVal, offset, WARP_SIZE);
    }

    if (wid == 0) {
        dstData[gid / WARP_SIZE] = sumVal;
    }
}

template<>
void cudaCallReduceSUMWarpCom<unsigned int>(const size_t nElements, const unsigned int* srcData, unsigned int* dstData) {
	const size_t WARP_SIZE = 32;
	const size_t BLOCK_SIZE = 1024;
	reduce_sum_warp_com<
				WARP_SIZE, unsigned int><<<
				(nElements + BLOCK_SIZE - 1) / BLOCK_SIZE,
				BLOCK_SIZE>>>(
				nElements,
				srcData,
				dstData);
}

// #endif

#ifdef USE_THRUST
template<typename T>
T reduce_sum_thrust(thrust::device_vector<T> src)
{
	return thrust::reduce(src.begin(), src.end());
}
#endif

#ifdef USE_CUB
template<size_t BLOCK_SIZE, typename T>
__global__ void reduce_sum_cub(const size_t nElements, const T* srcData, T* dstData) 
{
    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    typedef cub::BlockReduce<T, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage TempStorage;

    T sumVal = 0;
    if (gid < nElements) {
        sumVal = BlockReduce(TempStorage).Sum(srcData[gid]);
    }

    if (threadIdx.x == 0) {
        dstData[blockIdx.x] = sumVal;
    }
}

template<>
void cubCallReduceSUM(const size_t nElements, const unsigned int* srcData, unsigned int* dstData)
{
	const size_t BLOCK_SIZE = 1024;
	reduce_sum_cub<
				BLOCK_SIZE, unsigned int><<<
				(nElements + BLOCK_SIZE - 1) / BLOCK_SIZE,
				BLOCK_SIZE>>>(
				nElements,
				srcData,
				dstData);
}
#endif