#include "kernel.h"

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

__global__ void compareKernel(unsigned* vectors, int size, int length, long* result)
{
	long long modelVectorIdx = blockIdx.x * 1024 + threadIdx.x;
	long pairs = 0;

	for (int i = modelVectorIdx + 1; i < size; i++)
	{
		int mistakes = 0;

		for (int word = 0; word < length; word++)
		{
			__syncthreads();
			mistakes += compareWord(vectors[word * size + i], vectors[word * size + modelVectorIdx]);

			if (mistakes > 1)
				break;
		}

		if (mistakes <= 1)
			pairs++;
	}

	result[modelVectorIdx] = pairs;

	//long long compareVectorIdx = (blockIdx.x % blocksPerModel) * length * 1024 + threadIdx.x;
	//int mistakes = 0;
	//int zombie = 0;
	//int copySeries = 0;

	//__shared__ unsigned compareVectors[1024];
	//__shared__ unsigned modelVector[1024];
	//__shared__ int finished[1];

	//if (threadIdx.x == 0)
	//	*finished = 0;

	//for (size_t i = 0; i < length; i++)
	//{
	//	if (i % 1024 == 0) {
	//		__syncthreads();
	//		if (threadIdx.x + copySeries * 1024 < length)
	//			modelVector[threadIdx.x] = vectors[modelVectorIdx * length + threadIdx.x + copySeries * 1024];

	//		copySeries++;
	//	}

	//	if (zombie == 1)
	//		continue;

	//	if (compareVectorIdx < size && mistakes < 2) {
	//		compareVectors[threadIdx.x] = coalesced[i * size + compareVectorIdx];
	//		mistakes += compareWord(modelVector[i], compareVectors[threadIdx.x]);
	//	}

	//	if (compareVectorIdx >= size || mistakes > 1)
	//	{
	//		zombie = 1;
	//		atomicAdd(finished, 1);
	//	}

	//	if (*finished == 1024)
	//		break;
	//}

	//__syncthreads();
	//result[modelVectorIdx * size + compareVectorIdx] = mistakes <= 1 && compareVectorIdx < size&& compareVectorIdx != modelVectorIdx;
}