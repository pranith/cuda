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
#define MAX_ACCESSES_PER_ITERATION 200
#define MAX_TRIALS 5

__global__ void test_kernel(char *src, long max_idx)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while(tid < max_idx) {
		src[tid] = tid;
		tid += blockDim.x * gridDim.x;
	}

	for (long iter = 0; iter < NUM_ITERATIONS; iter++) {
		#include "defines.h";
	}
}

int main(int argc, char *argv[])
{
	char *data;
	int mem_size;
	long max_idx;
	float time_elapsed, total_time = 0;
	int trials;

	if (argc < 2) {
		fprintf(stderr, "Usage: %s <mem size(MB)>\n", argv[0]);
		exit(0);
	}

	mem_size = atoi(argv[1]);

	cudaEvent_t before, after;
	cudaEventCreate(&before);
	cudaEventCreate(&after);
	cudaMallocManaged(&data, MB(mem_size));

	for (trials = 0; trials < MAX_TRIALS; trials++) {
		cudaEventRecord(before, 0);
		test_kernel<<<BLOCKS_PER_SM, THREADS_PER_BLOCK>>>(data, max_idx);
		cudaDeviceSynchronize();
		max_idx = MB(mem_size);
		cudaEventRecord(after, 0);

		cudaEventSynchronize(before);
		cudaEventSynchronize(after);
		cudaEventElapsedTime(&time_elapsed, before, after);
		total_time += time_elapsed;
		if (trials == 0)
			total_time = 0;
	}
	std::cout << mem_size << "," << total_time/(trials-1) << "," <<
		total_time/((trials-1)*mem_size) << std::endl;
	cudaFree(data);

	return 0;
}
