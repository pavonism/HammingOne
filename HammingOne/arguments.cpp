#include "kernel.h"

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

void CheckArguments(int argc, char** argv) {

	if (argc < MIN_ARGUMENTS || argc > MAX_ARGUMENTS)
		ExitWithWrongNumberOfArgs(argv[0]);

	if (argc >= ARG_SHOW_PAIRS_NUMBER && strcmp(argv[ARG_SHOW_PAIRS_NUMBER], ARG_SHOW_PAIRS) != 0)
		ExitWithWrongArgs(argv[0], argv[2]);
}