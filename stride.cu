#include <cuda.h>
#include <stdlib.h>
#include <iostream>

#define THREADS_PER_BLOCK 32
#define BLOCKS_PER_SM 6

#define KB(x) ((x) << 10)
#define MB(x) ((x) << 20)

#define DIVIDE 1
#define CACHE_BLOCK_SIZE 64

__global__ void test_kernel(char *data, int maxIdx)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while(tid < maxIdx/DIVIDE) {
		data[tid] = tid;
		tid += blockDim.x * gridDim.x;
	}
#if (DIVIDE == 2)
	tid = threadIdx.x + blockIdx.x * blockDim.x;
	while(tid < maxIdx/2) {
		data[tid] = tid;
		tid += blockDim.x * gridDim.x;
	}
#endif
}

int main()
{
	char *data;

	for (int i = 1; i < 256; i++) {
		int maxIdx = MB(i);
		float time_elapsed, total_time = 0;
		cudaEvent_t before, after;
		cudaEventCreate(&before);
		cudaEventCreate(&after);
		cudaMallocManaged(&data, MB(i));

#if 1
		for (int j = 0; j < MB(i); j+=CACHE_BLOCK_SIZE)
			data[j] = j;
#endif
		for (int tries = 0; tries < 5; tries++) {
			cudaEventRecord(before, 0);
			test_kernel<<<BLOCKS_PER_SM, THREADS_PER_BLOCK>>>(data, maxIdx);
			cudaDeviceSynchronize();
			cudaEventRecord(after, 0);

			cudaEventSynchronize(before);
			cudaEventSynchronize(after);
			cudaEventElapsedTime(&time_elapsed, before, after);
			total_time += time_elapsed;
		}
		std::cout << i << "," << total_time/5 << "," << total_time/(5*i) << std::endl;
		cudaFree(data);
	}

	return 0;
}
