#include "data_operator.h"
#include <stdio.h>
#include <string>

#pragma region Private

void DataOperator::CheckCudaError(cudaError_t cudaStatus, char* reason) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, reason);
		exit(1);
	}
}

void DataOperator::SetDevice()
{
	auto cudaStatus = cudaSetDevice(0);
	CheckCudaError(cudaStatus, "SetDevice");
}

void DataOperator::Free() {
	FreeHost(vectors);
	FreeHost(results);
	FreeDevice(dev_vectors);
	FreeDevice(dev_coalesced);
	FreeDevice(dev_results);
}

template<class T>
void DataOperator::CopyToHost(T* source, T* destination, int size, int length)
{
	auto cudaStatus = cudaMemcpy(destination, source, size * sizeof(T) * length, cudaMemcpyDeviceToHost);
	CheckCudaError(cudaStatus, "CopyToHost");
}

template<class T>
void DataOperator::CopyToDevice(T* source, T* destination, int size, int length)
{
	auto cudaStatus = cudaMemcpy(destination, source, size * sizeof(T) * length, cudaMemcpyHostToDevice);
	CheckCudaError(cudaStatus, "CopyToHost");
}

template<class T>
void DataOperator::AllocateHost(T*& destination, int size, int length)
{
	destination = new T[(long long)size * length];
}

template<class T>
void DataOperator::AllocateDevice(T*& destination, int size, int length)
{
	auto cudaStatus = cudaMalloc((void**)&destination, (long long)size * sizeof(T) * length);
	CheckCudaError(cudaStatus, "AllocateDevice");
}

template<class T>
void DataOperator::FreeHost(T*& table)
{
	if (table == NULL)
		return;

	delete[] table;
	table = NULL;
}

template<class T>
void DataOperator::FreeDevice(T*& table)
{
	if (table == NULL)
		return;

	auto status = cudaFree(table);
	CheckCudaError(status, "FreeDevice");
	table = NULL;
}

template<class T>
void DataOperator::ClearTableOnHost(T* table, int size, int length)
{
	memset(table, 0, (long long)size * length * sizeof(T));
}

template<class T> void
DataOperator::CreateCoalescedData(T* table, int size, int length) {

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
	delete[] coalesced;
}

void DataOperator::AllocateVectors(int size, int length) {
	this->size = size;
	this->length = length;

	AllocateHost(vectors, size, length);
}

void DataOperator::ReadDataFromFile(char* path) {
	long vectorsCount = 0;
	long vectorLength = 0;
	long currentLength = 0;
	int vectorsIt = 0;
	FILE* file = fopen(path, "r");
	
	if (file == NULL)
		ExitWrongFile();

	// Read parameters
	auto ret = fscanf(file, "%d,%d%", &vectorsCount, &vectorLength);
	
	if(ret != 2)
		ExitWrongFile();

	while (fgetc(file) != '\n');

	AllocateVectors(vectorsCount, ceil((double)vectorLength / 32));
	char* currentVectorBits = new char[vectorLength + 1];

	for (int i = 0; i < vectorsCount; i++)
	{
		auto size = fread(currentVectorBits, sizeof(char), vectorLength + 1, file);

		if (size != vectorLength + 1)
			ExitWrongFile();

		for (currentLength = 0; currentLength < vectorLength; currentLength+=32)
		{
			uint_fast32_t word = 0;

			for (int bit = 0; bit < 32; bit++)
			{
				if (currentVectorBits[currentLength + bit] == '1')
					word = word | (1 << (32 - bit - 1));
			}
			this->vectors[vectorsIt++] = word;
		}

		int lastBits = currentLength % 32;
		if (lastBits != 0) {
			uint_fast32_t word = 0;

			for (int bit = 0; bit < lastBits; bit++)
			{
				if (currentVectorBits[currentLength - lastBits + bit] == '1')
					word = word | (1 << (32 - bit - 1));
			}

			this->vectors[vectorsIt++] = word;
		}
	}

	delete[] currentVectorBits;
	fclose(file);
}

void DataOperator::ExitWrongFile() {
	fprintf(stderr, MSG_WRONG_FILE_FORMAT);
	exit(1);
}
#pragma endregion Private

#pragma region Public
DataOperator::DataOperator(char* path) {

	vectors = NULL;
	results = NULL;
	dev_results = NULL;
	dev_vectors = NULL;
	dev_coalesced = NULL;
	pairs = NULL;
	dev_pairs = NULL;

	ReadDataFromFile(path);
}

void DataOperator::CopyToDevice() {
	AllocateHost(results, size, 1);
	AllocateDevice(dev_vectors, size, length);
	AllocateDevice(dev_coalesced, size, length);
	AllocateDevice(dev_results, size, 1);

	ClearTableOnHost(results, size, 1);
	CopyToDevice<long>(results, dev_results, size, 1);
	CopyToDevice(vectors, dev_vectors, size, length);
	CreateCoalescedData(vectors, size, length);
}

DataOperator::~DataOperator() {
	Free();
}

long DataOperator::CalculateResultsOnHost() {

	CopyToHost(dev_results, results, size, 1);

	int result = 0;
	for (size_t i = 0; i < size; i++)
	{
		result += results[i];
	}

	return result;
}

void DataOperator::AllocatePairs() {

	auto vectorPairsWordsCount = GetPairsLength();
	AllocateHost(pairs, size, vectorPairsWordsCount);
	ClearTableOnHost(pairs, size, vectorPairsWordsCount);
	AllocateDevice(dev_pairs, size, vectorPairsWordsCount);
	CopyToDevice(pairs, dev_pairs, size, vectorPairsWordsCount);
}

template<class T>
T DataOperator::GetBit(T data, int bit) {
	return (data >> bit) & 1;
}

void DataOperator::PrintVector(int indx) {

	for (size_t j = 0; j < length; j++)
	{
		for (size_t bit = 0; bit < 32; bit++)
		{
			printf("%d", (this->vectors[indx * length + j] >> (32 - bit - 1)) & 1);
		}
	}
}

void DataOperator::PrintPairs() {

	auto vectorPairsWordsCount = GetPairsLength();
	CopyToHost(dev_pairs, pairs, size, vectorPairsWordsCount);
	int pairs = 0;

	for (int vector = 0; vector < size; vector++)
	{
		for (size_t word = 0; word < vectorPairsWordsCount; word++)
		{
			for (size_t bit = 0; bit < WORD_BIT_LENGTH; bit++)
			{
				if (GetBit(this->pairs[vector * vectorPairsWordsCount + word], bit)) {
					printf("Pair %d:\n", pairs++);
					printf("Vector 1:\n");
					PrintVector(vector);
					printf("\n");
					printf("Vector 2:\n");
					PrintVector(word * WORD_BIT_LENGTH + bit);
					printf("\n");
				}
			}
		}
	}
}

void DataOperator::PrintVectors() {

	for (size_t i = 0; i < size; i++)
	{
		PrintVector(i);
		printf("\n");
	}
}

void DataOperator::CopyResults() {
	CopyToHost(dev_results, results, size, 1);
}

long DataOperator::GetSize() {
	return this->size;
}

long DataOperator::GetLength() {
	return this->length;
}

long DataOperator::GetPairsLength() {
	return ceil((double)this->size / sizeof(uint_fast32_t) / BYTE_LENGTH);
}
#pragma endregion Public