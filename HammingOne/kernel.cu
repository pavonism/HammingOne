#include "kernel.h"


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

int main(int argc, char** argv)
{
	CheckArguments(argc, argv);
	DataGenerator dataGenFromFile = DataGenerator(argv[1]);

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

