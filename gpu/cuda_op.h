#ifndef CUDA_OP_H
#define CUDA_OP_H

#include <cuda_runtime.h>

#ifdef USE_THRUST
#include <thrust/device_vector.h>	// memory
#include <thrust/reduce.h>			// op::reduce
#endif

#ifdef USE_CUB
#include <cub/block/block_reduce.cuh>	// op::reduce
#endif

// Reduce
template <typename T>
void cudaCallReduceSUMSharedMem(const size_t nElements, const T* srcData, T* dstData);

template <typename T>
void cudaCallReduceSUMWarpCom(const size_t nElements, const T* srcData, T* dstData);


#ifdef USE_THRUST
template<typename T>
T thrustCallReduceSUM(thrust::device_vector<T> src);
#endif

#ifdef USE_CUB
template <typename T>
void cubCallReduceSUM(const size_t nElements, const T* srcData, T* dstData);
#endif


// Eltwise

#endif