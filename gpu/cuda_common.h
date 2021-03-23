#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(condition) \
	/* Code block avoids redefinition of cudaError_t error */ \
	do { \
		cudaError_t error = condition; \
		if (error != cudaSuccess) { \
		  std::cout << cudaGetErrorString(error) << std::endl; \
		} \
	} while (0)

void SetGPUID(int device_id) {
	int current_device;
	CUDA_CHECK(cudaGetDevice(&current_device));
	if (current_device == device_id) {
		return;
	}
	// The call to cudaSetDevice must come before any calls to Get, which
	// may perform initialization using the GPU.
	CUDA_CHECK(cudaSetDevice(device_id));
}

#endif