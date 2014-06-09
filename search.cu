#include <cuda.h>
#include <assert.h>
#include <omp.h>
#include <iostream>

#define KB(x) ((x) << 10)
#define MB(x) ((x) << 20)

#define THREADS_PER_BLOCK 512

__global__ void SearchKernel(int *array, int search_key, int *search_result) 
{
	if (array[threadIdx.x + blockIdx.x * blockDim.x] == search_key)
		*search_result = 1;
}

void SearchLocal(int *array, int max_idx, int search_key, int *search_result)
{
	for (int i = 0; i < max_idx; i++)
		if (array[i] == search_key)
			*search_result = 1;
}

__managed__ int search_key, search_result = 0;
int main()
{
	int *array;
	int size = 1000 * sizeof(int);//MB(400);
	int max_idx = size / sizeof(int);

	std::cout << "Enter number to search " << std::endl;
	std::cin >> search_key;

	cudaMallocManaged(&array, size);

	for (int idx = 0; idx < max_idx; idx++)
		array[idx] = idx * idx;
	
	//#pragma omp sections
	{
		//#pragma omp section
		{
			dim3 numBlocks(max_idx / (2 * THREADS_PER_BLOCK), 1);
			SearchKernel<<<numBlocks, THREADS_PER_BLOCK>>>(array, search_key,	&search_result);
			cudaDeviceSynchronize();
		}
		//#pragma omp section
		SearchLocal(array+max_idx/2, max_idx/2, search_key, &search_result);
	}
	std::cout << search_result << std::endl;

	return 0;
}
