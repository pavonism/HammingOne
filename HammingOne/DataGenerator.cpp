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
	destination = new T[(long long)size * length];
}

template<class T>
void DataGenerator::AllocateDevice(T*& destination, int size, int length)
{
	auto cudaStatus = cudaMalloc((void**)&destination, (long long)size * sizeof(T) * length);
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
	memset(table, 0, (long long)size * length * sizeof(T));
}

void DataGenerator::ExitWrongFile() {
	fprintf(stderr, MSG_WRONG_FILE_FORMAT);
	exit(1);
}

DataGenerator::DataGenerator(int size, int length)
{
	vectors = GenerateRandomData(size, length);
	AllocateData(size, length);
}

void DataGenerator::AllocateData(int size, int length) {
	AllocateHost(results, size, size);
	AllocateDevice(dev_vectors, size, length);
	AllocateDevice(dev_coalesced, size, length);
	AllocateDevice(dev_results, size, size);

	vectors[0] = 123;
	vectors[1] = 123;
	vectors[1500] = 123;
	vectors[1501] = 123;


	ClearTableOnHost(results, size, size);
	CopyToDevice<bool>(results, dev_results, size, size);
	CopyToDevice(vectors, dev_vectors, size, length);
	CreateCoalescedData(vectors, size, length);
}

void DataGenerator::AllocateVectors(int size, int length) {
	this->size = size;
	this->length = length;

	AllocateHost(vectors, size, length);
}

void DataGenerator::PrintVectors() {

	for (size_t i = 0; i < 2; i++)
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

DataGenerator::DataGenerator(char* path) {

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

	printf("%ld, %ld\n", vectorsCount, vectorLength);
	AllocateVectors(vectorsCount, vectorLength / 32 + 1);
	char* currentVectorBits = new char[vectorLength + 1];
	memset(currentVectorBits, 0, (vectorLength + 1) * sizeof(char));
	seeker = i + 1;
	long long vectorsIt = 0;

	do {

		for (currentVector; currentVector < vectorsCount && seeker < readBytes;)
		{
			for (currentLength; currentLength < vectorLength && seeker < readBytes;)
			{
				if(buf[seeker] != 0x0D && buf[seeker] != 0x0A)
					currentVectorBits[currentLength++] = buf[seeker];
				seeker++;

				if (currentLength % 32 == 0 && currentLength > 0) {

					unsigned word = 0;

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
					unsigned word = 0;

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