#pragma once
#ifndef DataGenerator

#include "cuda_runtime.h"

#define MSG_WRONG_FILE_FORMAT "Wrong file format! Terminating...\n"
#define FILE_BUFFER_LENGTH 1024

class DataGenerator {
private:
	long size;
	long length;
	void CheckCudaError(cudaError_t cudaStatus, char* reason);
	void SetDevice();

	void Free();
	
	void ReadDataFromFile(char* path);
	void PrintVectors();
	void AllocateVectors(int size, int length);
	template<class T> void CopyToHost(T* source, T* destination, int size, int length);
	template<class T> void CopyToDevice(T* source, T* destination, int size, int length);
	template<class T> void AllocateHost(T*& destination, int size, int length);
	template<class T> void AllocateDevice(T*& destination, int size, int length);
	template<class T> void FreeHost(T*& table);
	template<class T> void FreeDevice(T*& table);
	template<class T> void ClearTableOnHost(T* table, int size, int length);
	template<class T> void CreateCoalescedData(T* table, int size, int length);
	void ExitWrongFile();
public: 
	unsigned* dev_coalesced;
	unsigned* dev_vectors;
	long* dev_results;
	unsigned* vectors;
	long* results;

	DataGenerator(int size, int length);
	DataGenerator(char* path);
	~DataGenerator();
	//DataGenerator(T* vectors, int size, int length);
	unsigned* GenerateRandomData(int size, int length);
	int CalculateResults();
	void CopyResults();
	void CopyToDevice();
	long GetSize();
	long GetLength();
};

#endif // !DataGenerator