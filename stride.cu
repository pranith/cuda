#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#define THREADS_PER_BLOCK 32
#define BLOCKS_PER_SM 1

#define KB(x) ((x) << 10)
#define MB(x) ((x) << 20)

#define CACHE_BLOCK_SIZE 64
#define NUM_ITERATIONS 10000000
#define ACCESSES_PER_ITERATION 200

__managed__ __device__ long max_idx = 0;

__global__ void test_kernel(char *src)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while(tid < max_idx) {
		data[tid] = tid;
		tid += blockDim.x * gridDim.x;
	}
}

int main(int argc, char *argv[])
{
	char *data;
	int mem_size;

	if (argc < 2) {
		fprintf(stderr, "Usage: %s <mem size(MB)>\n", argv[0]);
		exit(0);
	}

	mem_size = atoi(argv[1]);
	max_idx = MB(mem_size);

	float time_elapsed, total_time = 0;
	cudaEvent_t before, after;
	cudaEventCreate(&before);
	cudaEventCreate(&after);
	cudaMallocManaged(&data, MB(mem_size));

	for (int tries = 0; tries < 5; tries++) {
		cudaEventRecord(before, 0);
		test_kernel<<<BLOCKS_PER_SM, THREADS_PER_BLOCK>>>(data);
		cudaDeviceSynchronize();
		cudaEventRecord(after, 0);

		cudaEventSynchronize(before);
		cudaEventSynchronize(after);
		cudaEventElapsedTime(&time_elapsed, before, after);
		total_time += time_elapsed;
	}
	std::cout << mem_size << "," << total_time/5 << "," <<
		total_time/(5*mem_size) << std::endl;
	cudaFree(data);

	return 0;
}
