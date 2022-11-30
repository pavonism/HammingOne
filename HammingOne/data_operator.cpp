#include "data_operator.h"

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
	FILE* file = fopen(path, "r");
	char buf[FILE_BUFFER_LENGTH];
	long vectorsCount = 0;
	long vectorLength = 0;
	long currentVector = 0;
	long currentLength = 0;
	size_t readBytes;

	if (file == NULL)
		ExitWrongFile();

	// Read parameters 
	readBytes = fread(buf, sizeof(char), FILE_BUFFER_LENGTH, file);
	int seeker = 0;
	int i = 0;

	// Vectors count
	while (buf[i] != ',') i++;
	buf[i] = '\0';
	seeker = i + 1;
	vectorsCount = atol(buf);

	// Vectors length
	while (buf[i] != 0x0D && buf[i] != 0x0A) i++;
	buf[i] = '\0';
	vectorLength = atol(buf + seeker);

	//printf("%ld, %ld\n", vectorsCount, vectorLength);
	AllocateVectors(vectorsCount, ceil((double)vectorLength / 32));
	char* currentVectorBits = new char[vectorLength + 1];
	memset(currentVectorBits, 0, (vectorLength + 1) * sizeof(char));
	seeker = i + 1;
	long long vectorsIt = 0;

	do {

		for (currentVector; currentVector < vectorsCount && seeker < readBytes;)
		{
			for (currentLength; currentLength < vectorLength && seeker < readBytes;)
			{
				if (buf[seeker] != 0x0D && buf[seeker] != 0x0A)
					currentVectorBits[currentLength++] = buf[seeker];
				seeker++;

				if (currentLength % 32 == 0 && currentLength > 0) {

					uint_fast32_t word = 0;

					for (int bit = 0; bit < 32; bit++)
					{
						if (currentVectorBits[currentLength - 32 + bit] == '1')
							word = word | (1 << (32 - bit - 1));
					}

					this->vectors[vectorsIt++] = word;
				}
			}
			if (seeker < readBytes) {

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

				currentLength = 0;
				currentVector++;
				memset(currentVectorBits, 0, vectorLength * sizeof(char));
			}
		}

		seeker = 0;

	} while ((readBytes = fread(buf, sizeof(char), FILE_BUFFER_LENGTH, file)) > 0);

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

void DataOperator::PrintVectors() {

	for (size_t i = 0; i < size; i++)
	{
		for (size_t j = 0; j < length; j++)
		{
			for (size_t bit = 0; bit < 32; bit++)
			{
				printf("%d", (this->vectors[i * length + j] >> (32 - bit - 1)) & 1);

			}
		}

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
#pragma endregion Public