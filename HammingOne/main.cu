#include "main.h"

int main(int argc, char** argv)
{
	CheckArguments(argc, argv);
	arguments_t args = InitializeArguments(argc, argv);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	MeasureStart(&start, "Loading data from file...\n");
	DataOperator dataSource = DataOperator(args.filePath);
	MeasureStop(&start, &stop);

	MeasureStart(&start, "Copying data to device...\n");
	dataSource.CopyToDevice();
	MeasureStop(&start, &stop);

	if (args.showPairs)
		dataSource.AllocatePairs();
	
	int blockCount = ceil((double)dataSource.GetSize() / 1024);

	MeasureStart(&start, "Calculating on device...\n");
	compareKernel << <blockCount, 1024 >> > (dataSource.dev_coalesced, dataSource.GetSize(), dataSource.GetLength(), dataSource.dev_results, dataSource.dev_pairs, dataSource.GetPairsLength());
	auto cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching compareKernel!\n", cudaStatus);
	}
	MeasureStop(&start, &stop);

	MeasureStart(&start, "Calculating pairs...\n");
	long result = thrust::reduce(thrust::device, dataSource.dev_results, dataSource.dev_results + dataSource.GetSize(), 0);
	MeasureStop(&start, &stop);
	printf("Number of pairs: %d\n", result);

	if(args.showPairs)
		dataSource.PrintPairs();

	if (args.showCPUOutput) {
		MeasureStart(&start, "Calculating pairs on host...\n");
		result = CheckHostResult(dataSource.vectors, dataSource.GetSize(), dataSource.GetLength());
		MeasureStop(&start, &stop);
		printf("Number of pairs: %d\n", result);
	}
	
	return EXIT_SUCCESS;
}