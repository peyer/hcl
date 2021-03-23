#include "cuda_common.h"
#include "cuda_op.h"

int main(int argc, char** argv)
{	
	if (argc < 2) {
		printf("Usage: %s GPU_ID\n", argv[0]);
		return -1;
	}
	const int gpu_id = atoi(argv[1]);
	SetGPUID(gpu_id);
	
	const size_t
		n = 1 << 30,
		BLOCK_SIZE = 1 << 10,
		WARP_SIZE = 1 << 5,
		REDUCE_SIZE = (n + WARP_SIZE - 1) / WARP_SIZE;
	thrust::device_vector<unsigned> src(n, 1), tmp(REDUCE_SIZE);
	const unsigned char opDesc[4][128] = {
	"======thrust::reduce=======", 
	"======shared_sum_kernel=======", 
	"======warp_primitive_sum_kernel=======", 
	"======cub::BlockReduce reduce_sum_cub======="};
	for (int op = 0; op < 4; ++op) {
		unsigned sum;
		cudaEvent_t beg, end;
		cudaEventCreate(&beg);
		cudaEventCreate(&end);
		cudaEventRecord(beg, 0);
		if (op == 0) {
			sum = thrust::reduce(src.begin(), src.begin() + n);
		}
		if (op == 1) {
			cudaCallReduceSUMSharedMem(n, thrust::raw_pointer_cast(src.data()), thrust::raw_pointer_cast(tmp.data()));
			sum = thrust::reduce(tmp.begin(), tmp.begin() + (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
		}
		if (op == 2) {
			cudaCallReduceSUMWarpCom(n, thrust::raw_pointer_cast(src.data()), thrust::raw_pointer_cast(tmp.data()));
			sum = thrust::reduce(tmp.begin(), tmp.begin() + (n + WARP_SIZE - 1) / WARP_SIZE);
		}
		if (op == 3) {
			cubCallReduceSUM(n, thrust::raw_pointer_cast(src.data()), thrust::raw_pointer_cast(tmp.data()));
			sum = thrust::reduce(tmp.begin(), tmp.begin() + (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
		}
		cudaEventRecord(end, 0);
		cudaEventSynchronize(beg);
		cudaEventSynchronize(end);
		float elapsed_time;
		cudaEventElapsedTime(
			&elapsed_time,
			beg,
			end);
		std::cout << opDesc[op] << std::endl;
		std::cout << sum << ": " << elapsed_time << " ms elapsed." << std::endl;
		std::cout << std::endl;
		// printf("%u : %fms elapsed.\n", sum, elapsed_time);
	}
	
	return 0;
}
