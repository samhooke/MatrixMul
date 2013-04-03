#include "cutilmk2.h"

CUTBoolean cutReadFilef(const char* filename, float** data, unsigned int* len, bool verbose) {
	return sdkReadFile(filename, &data, len, verbose) ? CUTTrue : CUTFalse;
}

CUTBoolean cutReadFilei(const char* filename, int** data, unsigned int* len, bool verbose) {
	return sdkReadFile(filename, &data, len, verbose) ? CUTTrue : CUTFalse;
}

CUTBoolean cutWriteFilef(const char* filename, const float* data, unsigned int len, const float epsilon, bool verbose) {
	return sdkWriteFile(filename, &data, len, epsilon, verbose, false) ? CUTTrue : CUTFalse;
}

CUTBoolean cutComparefe(const float* reference, const float* data, const unsigned int len, const float epsilon ) {
	return compareData(reference, data, len, epsilon, 0.1f) ? CUTTrue : CUTFalse;
}
