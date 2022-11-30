#include "main.h"

void CheckArguments(int argc, char** argv) {

	if (argc < MIN_ARGUMENTS || argc > MAX_ARGUMENTS)
		ExitWithWrongNumberOfArgs(argv[0]);

	if (argc >= ARG_SHOW_PAIRS_NUMBER && strcmp(argv[ARG_SHOW_PAIRS_NUMBER], ARG_SHOW_PAIRS) != 0)
		ExitWithWrongArgs(argv[0], argv[ARG_SHOW_PAIRS_NUMBER]);

	if (argc >= ARG_SHOW_CPU_NUMBER && strcmp(argv[ARG_SHOW_CPU_NUMBER], ARG_SHOW_CPU) != 0)
		ExitWithWrongArgs(argv[0], argv[ARG_SHOW_CPU_NUMBER]);
}

arguments_t InitializeArguments(int argc, char** argv) {

	arguments_t args = { argv[1], false, false};

	if (argc < 3)
		return args;
	args.showPairs = true;
	
	if (argc < 4)
		return args;

	args.showCPUOutput = true;

	return args;
}

void PrintUsage(FILE* dest, char* programName) {
	fprintf(dest, MSG_DESC);
	fprintf(dest, MSG_USAGE, programName);
}

void ExitByReason(char* programName, char* message) {
	fprintf(stderr, message);
	PrintUsage(stderr, programName);
	exit(1);
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