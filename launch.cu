#include <cuda.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 32
#define BLOCKS_PER_SM 6

#define KB(x) ((x) << 10)
#define MB(x) ((x) << 20)

__global__ void test_kernel(int maxIdx)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while(tid < maxIdx) {
		data[tid] = tid;
		tid += blockDim.x * gridDim.x;
	}
}

int main()
{
	char *data;

	for (int i = 1; i < 256; i++) {
		int maxIdx = MB(i);
		data = cudaMallocManaged(MB(i));

		test_kernel<<<BLOCKS_PER_SM, THREADS_PER_SM>>>(maxIdx);
		cudaDeviceSynchronize();
	}
}

