#pragma once
#ifndef KERNEL

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
#include "DataGenerator.h"

#pragma region Constants
#define MAX_ARGUMENTS 3
#define MIN_ARGUMENTS 2

#define MSG_WRONG_ARG "Wrong argument: %s\n"
#define MSG_WRONG_ARG_NUMBER "Wrong number of arguments!\n"
#define MSG_DESC "Program finds pairs of vectors with Hamming distance equals one\n"
#define MSG_USAGE "\
Usage: %s <path> <optional: show-pairs>\n\
path - path to a file with vectors \n\
show-pairs - as a result show whole vectors, not only number of pairs\n"

#define ARG_SHOW_PAIRS "show-pairs"
#define ARG_SHOW_PAIRS_NUMBER 3
#pragma endregion Constants

#pragma region Checking Arguments
void PrintUsage(FILE* , char* );
void ExitByReason(char* programName, char* message);
void ExitWithWrongNumberOfArgs(char* programName);
void ExitWithWrongArgs(char* programName, char* wrongArg);
void CheckArguments(int argc, char** argv);
#pragma endregion Checking Arguments

#pragma region CUDA Functionality 
template<class T> __host__ __device__ T getBit(T data, int bit);
__host__ __device__ bool IsPowerOfTwo(unsigned word);
__host__ __device__ int compareWord(unsigned first, unsigned second);
__global__ void compareKernel(unsigned* vectors, int size, int length, long* result);
#pragma endregion CUDA Functionality 

#endif // !KERNEL