#pragma once
#include "cuda_runtime.h"


class DataGenerator {
private:
	int size;
	int length;
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
public: 
	unsigned* dev_coalesced;
	unsigned* dev_vectors;
	bool* dev_results;
	unsigned* vectors;
	bool* results;

	DataGenerator(int size, int length);
	~DataGenerator();
	//DataGenerator(T* vectors, int size, int length);
	unsigned* GenerateRandomData(int size, int length);
	int CalculateResults();
};


