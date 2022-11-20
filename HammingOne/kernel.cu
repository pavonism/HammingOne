#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <stdio.h>
#include "DataGenerator.h"
#include <cmath>

template<class T>
__host__ __device__ T getBit(T data, int bit) {
	return (data >> bit) & 1;
}

__host__ __device__ int compareWord(unsigned first, unsigned second) {

	int counter = 0;

	for (int j = 0; j < sizeof(unsigned) * 8; j++)
	{
		if (getBit(first, j) != getBit(second, j))
		{
			counter++;
		}
	}

	return counter;
}

__global__ void compareKernel(unsigned* vectors, unsigned* coalesced, int size, int length, bool* result)
{
	int blocksPerModel = size / 1024 + 1;
	int modelVectorIdx = (double)blockIdx.x / blocksPerModel;
	int compareVectorIdx = (blockIdx.x % blocksPerModel) * length * 1024 + threadIdx.x;
	int mistakes = 0;
	int copySeries = 0;

	__shared__ unsigned compareVectors[1024];
	__shared__ unsigned modelVector[1024];

	for (size_t i = 0; i < length; i++)
	{
		if (i % 1024 == 0) {

			if(threadIdx.x + copySeries * 1024 < length)
				modelVector[threadIdx.x] = vectors[modelVectorIdx * length + threadIdx.x + copySeries*1024];

			copySeries++;
			__syncthreads();
		}

		if (compareVectorIdx < size && mistakes < 2) {
			compareVectors[threadIdx.x] = coalesced[i * size + compareVectorIdx];
			mistakes += compareWord(modelVector[i], compareVectors[threadIdx.x]);
		}

		__syncthreads();
	}

	result[modelVectorIdx * size + compareVectorIdx] = mistakes <= 1 && compareVectorIdx < size && compareVectorIdx != modelVectorIdx;
}

void CheckHostResult(unsigned* data, int DATA_SIZE, int DATA_LENGTH) {
	printf("HOST:\n");
	int result = 0;

	for (size_t i = 0; i < DATA_SIZE; i++)
	{
		for (size_t j = 0; j < DATA_SIZE; j++)
		{
			if (i <= j)
				continue;

			int counter = 0;

			for (size_t k = 0; k < DATA_LENGTH; k++)
			{
				counter += compareWord(data[i * DATA_LENGTH + k], data[j * DATA_LENGTH + k]);
				if (counter > 1)
					break;
			}

			if (counter < 2)
				result++;
		}
	}

	printf("Liczba par na hoście: %d\n", result);
}


int main()
{
    const int DATA_LENGTH = 1;
    const int DATA_SIZE = 20000;
	DataGenerator dataGenerator = DataGenerator(DATA_SIZE, DATA_LENGTH);
	auto data = dataGenerator.vectors;
	int blockCount = DATA_SIZE * (DATA_SIZE / 1024 + 1);


    compareKernel<<<blockCount, 1024>>>(dataGenerator.dev_vectors, dataGenerator.dev_coalesced, DATA_SIZE, DATA_LENGTH, dataGenerator.dev_results);
	
	// Check for any errors launching the kernel
	auto cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching compareKernel!\n", cudaStatus);
	}

	printf("Liczba par na device: %d\n", dataGenerator.CalculateResults());

	CheckHostResult(data, DATA_SIZE, DATA_LENGTH);

    return 0;
}

