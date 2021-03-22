
// NULL, work-item = {(N + 3) / 4, (M + 3) / 4}, NULL
// AT(K * M), B(K * N)
__kernel void gemm_at_bn_buffer(const int M, const int K, const int N,
	__read_only float* A, __read_only float* B, __write_only float* C)
{
	const int ix = get_global_id(0) << 2;
	const int iy = get_global_id(1) << 2;
	
	if (ix >= N || iy >= M) return;
	
	float4 sumVal[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        sumVal[i] = (float4)(0.0f);
    }

	for (int k = 0; k < K; k++) {
		float4 aLoad = vload4(0, A + k * M + iy);
		float4 bLoad = vload4(0, B + k * N + ix);
		
		sumVal[0] += aLoad.x * bLoad;
		sumVal[1] += aLoad.y * bLoad;
		sumVal[2] += aLoad.z * bLoad;
		sumVal[3] += aLoad.w * bLoad;
	}
	
	vstore4(C, 0, ix * N + iy);
	vstore4(C, 0, ix * N + iy + 1);
	vstore4(C, 0, ix * N + iy + 2);
	vstore4(C, 0, ix * N + iy + 3);
}

// AT (h = K, w = M / 4, channel = RGBA, data_type=float), B(h = K, w = N / 4, channel = RGBA, data_type=float)
// C  (h = M, w = N / 4, channel = RGBA, data_type=float)
__const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
__kernel void gemm_at_bn_image2d(const int M, const int K, const int N,
	__read_only image2d_t A, __read_only image2d_t B, __write_only image2d_t C)
{
	const int ix = get_global_id(0);
	const int iy = get_global_id(1);
	
	if (ix << 2 >= N || iy << 2 >= M) return;
	
	float4 sumVal[4]; 
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        sumVal[i] = (float4)(0.0f);
    }
	
	for (int k = 0; k < K; k++) {
		float4 aLoad = read_imagef(A, sampler, int2(iy, k);
		float4 bLoad = read_imagef(B, sampler, int2(ix, k);
		
		sumVal[0] += aLoad.x * bLoad;
		sumVal[1] += aLoad.y * bLoad;
		sumVal[2] += aLoad.z * bLoad;
		sumVal[3] += aLoad.w * bLoad;
	}
	
    iy <<= 2;
	write_imagef(C, int2(ix, iy), sumVal[0]);
	write_imagef(C, int2(ix, iy + 1), sumVal[1]);
	write_imagef(C, int2(ix, iy + 2), sumVal[2]);
	write_imagef(C, int2(ix, iy + 3), sumVal[3]);
}