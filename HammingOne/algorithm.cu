#include "main.h"

__host__ __device__ bool IsPowerOfTwo(uint_fast32_t word) {

	return word && !(word & (word - 1));
}

__host__ __device__ int compareWord(uint_fast32_t first, uint_fast32_t second) {

	uint_fast32_t comparison = first ^ second;

	if ((comparison) == 0)
		return 0;
	if (IsPowerOfTwo(comparison))
		return 1;

	return 2;
}

__global__ void compareKernel(uint_fast32_t* vectors, int size, int length, long* result)
{
	long long modelVectorIdx = blockIdx.x * 1024 + threadIdx.x;
	long pairs = 0;


	if(modelVectorIdx < size)
		for (int i = 0; i < modelVectorIdx; i++)
		{
			int mistakes = 0;

			for (int word = 0; word < length; word++)
			{
				mistakes += compareWord(vectors[word * size + i], vectors[word * size + modelVectorIdx]);

				if (mistakes > 1)
					break;
			}

			if (mistakes == 1)
				pairs++;
		}

	result[modelVectorIdx] = pairs;
}

__host__ int CheckHostResult(uint_fast32_t* data, int DATA_SIZE, int DATA_LENGTH) {
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

	return result;
}