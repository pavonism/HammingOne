#include "DataGenerator.h"
#include <random>

char** DataGenerator::GenerateRandomData(int size, int length) {

	if (size <= 0 || length <= 0)
		return nullptr;

	char** data = new char*[size];
	std::random_device engine;
	unsigned randomBytes;

	for (size_t i = 0; i < size; i++)
	{
		data[i] = new char[length];

		for (size_t j = 0; j < length; j++)
		{
			randomBytes = engine();
			data[i][j] = *((char*)&randomBytes);
		}
	}

	return data;
}
