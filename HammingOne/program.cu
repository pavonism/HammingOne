#include "main.h"

void CheckArguments(int argc, char** argv) {

	if (argc < MIN_ARGUMENTS || argc > MAX_ARGUMENTS)
		ExitWithWrongNumberOfArgs(argv[0]);
}

arguments_t InitializeArguments(int argc, char** argv) {

	arguments_t args = { argv[1], false, false };

	for (size_t i = 2; i < argc; i++)
	{
		if (strcmp(argv[i], ARG_SHOW_PAIRS) == 0) {
			args.showPairs = true;
		}
		else if (strcmp(argv[i], ARG_SHOW_CPU) == 0) {
			args.showCPUOutput = true;
		}
		else {
			ExitWithWrongArgs(argv[0], argv[i]);
		}
	}


	return args;
}

void PrintUsage(FILE* dest, char* programName) {
	fprintf(dest, MSG_DESC);
	fprintf(dest, MSG_USAGE, programName);
}

void ExitByReason(char* programName, char* message) {
	fprintf(stderr, message);
	PrintUsage(stderr, programName);
	exit(EXIT_FAILURE);
}

void ExitWithWrongNumberOfArgs(char* programName) {
	ExitByReason(programName, MSG_WRONG_ARG_NUMBER);
}

void ExitWithWrongArgs(char* programName, char* wrongArg) {
	fprintf(stderr, MSG_WRONG_ARG, wrongArg);
	PrintUsage(stderr, programName);
	exit(1);
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