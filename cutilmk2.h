/*
 * cutil.h was superseded by helper_functions.h
 * This code acts as a simple layer between the two
 *
 * These functions from cutil.h:
 * - cutReadFilef
 * - cutReadFilei
 * - cutWriteFilef
 * - cutComparefe
 * Have been replaced by these functions from helper_functions.h:
 * - sdkReadFile
 * - sdkWriteFile
 * - compareData
 */

#include <helper_functions.h>

enum CUTBoolean
{
	CUTFalse = 0,
	CUTTrue = 1
};

CUTBoolean cutReadFilef(const char* filename, float** data, unsigned int* len, bool verbose = false);
CUTBoolean cutReadFilei(const char* filename, int** data, unsigned int* len, bool verbose = false);
CUTBoolean cutWriteFilef(const char* filename, const float* data, unsigned int len, const float epsilon, bool verbose = false);
CUTBoolean cutComparefe(const float* reference, const float* data, const unsigned int len, const float epsilon );
