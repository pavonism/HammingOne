#pragma once
#ifndef KERNEL
#define KERNEL

// Cuda Headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

// System Headers
#include <stdio.h>
#include <cstdio>
#include <stdlib.h>
#include <cmath>
#include <string.h>

// Local Headers
#include "data_operator.h"

#pragma region Constants
#define MAX_ARGUMENTS 3
#define MIN_ARGUMENTS 2

#define MSG_WRONG_ARG "Wrong argument: %s\n"
#define MSG_WRONG_ARG_NUMBER "Wrong number of arguments!\n"
#define MSG_DESC "Program finds pairs of vectors with Hamming distance equals one\n"
#define MSG_USAGE "\
Usage: %s <path> <optional: show-pairs> <optional: show-cpu>\n\
path - path to a file with vectors \n\
show-pairs - as a result show whole vectors, not only number of pairs\n\
show-cpu - shows also a result calculated on host\n"

#define ARG_SHOW_PAIRS "show-pairs"
#define ARG_SHOW_CPU "show-cpu"
#define ARG_SHOW_PAIRS_NUMBER 3
#define ARG_SHOW_CPU_NUMBER 4
#pragma endregion Constants

#pragma region Program Logic
typedef struct arguments {
	char* filePath;
	bool showPairs;
	bool showCPUOutput;
} arguments_t;

void CheckArguments(int argc, char** argv);
arguments_t InitializeArguments(int argc, char** argv);
void PrintUsage(FILE* , char* );
void ExitByReason(char* programName, char* message);
void ExitWithWrongNumberOfArgs(char* programName);
void ExitWithWrongArgs(char* programName, char* wrongArg);
#pragma endregion Program Logic

#pragma region Measuring Time
void MeasureStart(cudaEvent_t* start, char* msg);
void MeasureStop(cudaEvent_t* start, cudaEvent_t* stop);
#pragma endregion Measuring Time

#pragma region CUDA Functionality 
__host__ __device__ bool IsPowerOfTwo(uint_fast32_t word);
__host__ __device__ int compareWord(uint_fast32_t first, uint_fast32_t second);
__global__ void compareKernel(uint_fast32_t* vectors, int size, int length, long* result);
__host__ int CheckHostResult(uint_fast32_t* data, int size, int length);
#pragma endregion CUDA Functionality 

#endif // !KERNEL