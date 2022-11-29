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

__host__ __device__ bool IsPowerOfTwo(unsigned word) {

	return word && !(word & (word - 1));
}

__host__ __device__ int compareWord(unsigned first, unsigned second) {

	unsigned comparison = first ^ second;

	if ((comparison) == 0)
		return 0;
	if (IsPowerOfTwo(comparison))
		return 1;

	return 2;
}

__global__ void compareKernel(unsigned* vectors, unsigned* coalesced, int size, int length, bool* result)
{
	int blocksPerModel = size / 1024 + 1;
	long long modelVectorIdx = (double)blockIdx.x / blocksPerModel;
	long long compareVectorIdx = (blockIdx.x % blocksPerModel) * length * 1024 + threadIdx.x;
	int mistakes = 0;
	int zombie = 0;
	int copySeries = 0;

	__shared__ unsigned compareVectors[1024];
	__shared__ unsigned modelVector[1024];
	__shared__ int finished[1];

	if (threadIdx.x == 0)
		*finished = 0;

	for (size_t i = 0; i < length; i++)
	{
		if (i % 1024 == 0) {
			__syncthreads();
			if(threadIdx.x + copySeries * 1024 < length)
				modelVector[threadIdx.x] = vectors[modelVectorIdx * length + threadIdx.x + copySeries*1024];

			copySeries++;
		}

		if (zombie == 1)
			continue;

		if (compareVectorIdx < size && mistakes < 2) {
			compareVectors[threadIdx.x] = coalesced[i * size + compareVectorIdx];
			mistakes += compareWord(modelVector[i], compareVectors[threadIdx.x]);
		}

		if (compareVectorIdx >= size || mistakes > 1)
		{
			zombie = 1;
			atomicAdd(finished, 1);
		}

		if (*finished == 1024)
			break;
	}

	__syncthreads();
	result[modelVectorIdx * size + compareVectorIdx] = mistakes <= 1 && compareVectorIdx < size && compareVectorIdx != modelVectorIdx;
}

void CheckHostResult(unsigned* data, int DATA_SIZE, int DATA_LENGTH) {
	printf("HOST:\n");
	int result = 0;

	for (size_t i = 0; i < DATA_SIZE; i++)
	{
		for (size_t j = i + 1; j < DATA_SIZE; j++)
		{
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
    const int DATA_LENGTH = 1/*1000 / 32*/;
    const int DATA_SIZE = 50000;
	DataGenerator dataGenerator = DataGenerator(DATA_SIZE, DATA_LENGTH);
	auto data = dataGenerator.vectors;
	int blockCount = DATA_SIZE * (DATA_SIZE / 1024 + 1);
	cudaEvent_t start, stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	printf("Rozpoczynanie obliczeń na GPU...\n");
	cudaEventRecord(start, 0);
    compareKernel<<<blockCount, 1024>>>(dataGenerator.dev_vectors, dataGenerator.dev_coalesced, DATA_SIZE, DATA_LENGTH, dataGenerator.dev_results);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

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

	printf("Obliczanie zakończone...\n");
	printf("%f s\n", time / 1000);
	printf("Liczba par na device: %d\n", dataGenerator.CalculateResults());

	CheckHostResult(data, DATA_SIZE, DATA_LENGTH);

    return 0;
}

