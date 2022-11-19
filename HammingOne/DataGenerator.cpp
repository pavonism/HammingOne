#include "DataGenerator.h"
#include <random>

void DataGenerator::CheckCudaError(cudaError_t cudaStatus, char* reason) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, reason);
		exit(1);
	}
}

void DataGenerator::SetDevice()
{
	auto cudaStatus = cudaSetDevice(0);
	CheckCudaError(cudaStatus, "SetDevice");
}

void DataGenerator::Free() {
	FreeHost(vectors);
	FreeHost(results);
	FreeDevice(dev_vectors);
	FreeDevice(dev_coalesced);
	FreeDevice(dev_results);
}

template<class T>
void DataGenerator::CopyToHost(T* source, T* destination, int size, int length)
{
	auto cudaStatus = cudaMemcpy(destination, source, size * sizeof(T) * length, cudaMemcpyDeviceToHost);
	CheckCudaError(cudaStatus, "CopyToHost");
}

template<class T>
void DataGenerator::CopyToDevice(T* source, T* destination, int size, int length)
{
	auto cudaStatus = cudaMemcpy(destination, source, size * sizeof(T) * length, cudaMemcpyHostToDevice);
	CheckCudaError(cudaStatus, "CopyToHost");
}

template<class T>
void DataGenerator::AllocateHost(T*& destination, int size, int length)
{
	destination = new T[size * length];
}

template<class T>
void DataGenerator::AllocateDevice(T*& destination, int size, int length)
{
	auto cudaStatus = cudaMalloc((void**)&destination, size * sizeof(T) * length);
	CheckCudaError(cudaStatus, "AllocateDevice");
}

template<class T>
void DataGenerator::FreeHost(T*& table)
{
	if (table == nullptr)
		return;

	delete[] table;
	table = nullptr;
}

template<class T>
void DataGenerator::FreeDevice(T*& table)
{
	if (table == nullptr)
		return;

	auto status = cudaFree(table);
	CheckCudaError(status, "FreeDevice");
	table = nullptr;
}

template<class T>
void DataGenerator::ClearTableOnHost(T* table, int size, int length)
{
	memset(table, 0, size * length * sizeof(T));
}



DataGenerator::DataGenerator(int size, int length)
{
	vectors = GenerateRandomData(size, length);
	this->size = size;
	this->length = length;

	AllocateHost(results, size, size);
	AllocateDevice(dev_vectors, size, length);
	AllocateDevice(dev_coalesced, size, length);
	AllocateDevice(dev_results, size, size);

	vectors[0] = 123;
	vectors[1] = 123;

	ClearTableOnHost(results, size, size);
	CopyToDevice<bool>(results, dev_results, size, size);
	CopyToDevice(vectors, dev_vectors, size, length);
	CreateCoalescedData(vectors, size, length);
}

DataGenerator::~DataGenerator() {
	Free();
}

unsigned* DataGenerator::GenerateRandomData(int size, int length) {

	if (size <= 0 || length <= 0)
		return nullptr;

	unsigned* data = new unsigned[size * length];
	std::random_device engine;
	unsigned randomBytes;

	for (size_t i = 0; i < size; i++)
	{
		for (size_t j = 0; j < length; j++)
		{
			randomBytes = engine();
			data[i * length + j] = randomBytes;
		}
	}

	return data;
}

template<class T> void 
DataGenerator::CreateCoalescedData(T* table, int size, int length) {

	T* coalesced = new T[size * length];
	int counter = 0;

	for (size_t j = 0; j < length; j++)
	{
		for (size_t i = 0; i < size; i++)
		{
			coalesced[counter++] = table[i * length + j];
		}
	}

	CopyToDevice(coalesced, dev_coalesced, size, length);
}

int DataGenerator::CalculateResults() {

	CopyToHost(dev_results, results, size, size);

	int result = 0;
	for (size_t i = 0; i < size; i++)
	{
		for (size_t j = 0; j < size; j++)
		{
			if (i < j && results[i * size + j])
				result++;
		}
	}

	return result;
}