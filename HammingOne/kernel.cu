
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "DataGenerator.h"

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int length);

__host__ __device__ char getBit(char data, int bit) {
    return (data >> bit) & 1;
}

__host__ __device__ bool compareData(char* firstVector, char* secondVector, int length) {

    bool differ = false;

    for (size_t i = 0; i < length; i++)
    {
        for (size_t j = 0; j < 8; j++)
        {
            if (getBit(firstVector[i], j) != getBit(secondVector[i], j))
            {
                if (differ)
                    return false;
                differ = true;
            }
        }
    }

    return true;
}

__global__ void compareKernel(char** vectors, int length, bool** result)
{
    char* firstVector = vectors[blockDim.x];
    char* secondVector = vectors[threadIdx.x];

    result[blockDim.x][threadIdx.x] = compareData(firstVector, secondVector, length);
}

__host__ void printData(char* data, int lenght) {

    for (size_t i = 0; i < lenght; i++)
    {
        printf("%c", data[i]);
    }

    printf("\n");
}

__host__ void printComparison(char* firstVector, char* secondVector, int length) {
    printf("Pierwszy wektor: \n");
    printData(firstVector, length);
    printf("Drugi wektor: \n");
    printData(secondVector, length);
    printf("\n");
}

int main()
{
    const int DATA_LENGTH = 4;
    const int DATA_SIZE = 20000;
    char** data = DataGenerator::GenerateRandomData(DATA_SIZE, DATA_LENGTH);

    for (size_t i = 0; i < DATA_SIZE; i++)
    {
        for (size_t j = 0; j < DATA_SIZE; j++)
        {
            if (i == j)
                continue;

            if (compareData(data[i], data[j], DATA_LENGTH)) {
                printComparison(data[i], data[j], DATA_LENGTH);
            }
        }
    }

    return 0;
}

