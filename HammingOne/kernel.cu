#include "kernel.h"

void CheckHostResult(unsigned* data, int DATA_SIZE, int DATA_LENGTH) {
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

	printf("Number of pairs on host: %d\n", result);
}

void MeasureStart(cudaEvent_t* start, char* msg) {
	printf(msg);
	cudaEventRecord(*start, 0);
}

void MeasureStop(cudaEvent_t* start, cudaEvent_t* stop) {
	float time;
	
	cudaEventRecord(*stop, 0);
	cudaEventSynchronize(*stop);
	cudaEventElapsedTime(&time, *start, *stop);
	printf("%f s\n", time / 1000);
}

int main(int argc, char** argv)
{
	//CheckArguments(argc, argv);

	char* file = "C:\\Users\\spawl\\Desktop\\MiNI_5\\GPU\\HammingOne\\input.txt";

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	MeasureStart(&start, "Loading data from file...\n");
	DataGenerator dataSource = DataGenerator(file);
	MeasureStop(&start, &stop);

	MeasureStart(&start, "Copying data to device...\n");
	dataSource.CopyToDevice();
	MeasureStop(&start, &stop);

	int blockCount = ceil((double)dataSource.GetSize() / 1024);

	MeasureStart(&start, "Calculating on device...\n");
	compareKernel<<<blockCount, 1024>>>(dataSource.dev_coalesced, dataSource.GetSize(), dataSource.GetLength(), dataSource.dev_results);
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
	MeasureStop(&start, &stop);

	MeasureStart(&start, "Copying results to host...\n");
	auto hostResults = dataSource.CalculateResults();
	MeasureStop(&start, &stop);
	printf("[HOST] Number of pairs: %d\n", hostResults);

	MeasureStart(&start, "Calculating pairs...\n");
	long result = thrust::reduce(thrust::host, dataSource.results, dataSource.results + dataSource.GetSize(), 0);
	MeasureStop(&start, &stop);
	printf("Number of pairs: %d\n", result);

	MeasureStart(&start, "Calculating pairs on host...\n");
	CheckHostResult(dataSource.vectors, dataSource.GetSize(), dataSource.GetLength());
	MeasureStop(&start, &stop);
}

