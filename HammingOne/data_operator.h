#pragma once
#ifndef DATA_OPERATOR
#define DATA_OPERATOR
#include "cuda_runtime.h"
#include <random>

#define MSG_WRONG_FILE_FORMAT "Wrong file format! Terminating...\n"
#define FILE_BUFFER_LENGTH 1024
#define BYTE_LENGTH 8

class DataOperator {
private:
	long size;
	long length;

	void CheckCudaError(cudaError_t cudaStatus, char* reason);
	void SetDevice();
	void Free();

	template<class T> void CopyToHost(T* source, T* destination, int size, int length);
	template<class T> void CopyToDevice(T* source, T* destination, int size, int length);
	template<class T> void AllocateHost(T*& destination, int size, int length);
	template<class T> void AllocateDevice(T*& destination, int size, int length);
	template<class T> void FreeHost(T*& table);
	template<class T> void FreeDevice(T*& table);
	template<class T> void ClearTableOnHost(T* table, int size, int length);
	template<class T> void CreateCoalescedData(T* table, int size, int length);
	void AllocateVectors(int size, int length);
	void ExitWrongFile();
	void ReadDataFromFile(char* path);
public: 
	uint_fast32_t* dev_coalesced;
	uint_fast32_t* dev_vectors;
	long* dev_results;
	uint_fast32_t* vectors;
	long* results;

	DataOperator(char* path);
	void CopyToDevice();
	~DataOperator();
	long CalculateResultsOnHost();
	void PrintVectors();
	void CopyResults();
	long GetSize();
	long GetLength();
};
#endif // !DATA_OPERATOR