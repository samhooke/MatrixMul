/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a Non-Disclosure Agreement.  Any reproduction or
 * disclosure to any third party without the express written consent of
 * NVIDIA is prohibited.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */
// Modified by Sam Hooke, April 2013

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "cutilmk2.h" // Replaces cutil.h
#include "matrixmul_kernel.h"

//@@ Choose which kernel to use
// Kernel 1: Works with any sized matrices that have dimensions
//           that are multiples of BLOCK_SIZE, and bigger than or
//           equal to BLOCK_SIZE x BLOCK_SIZE.
// Kernel 2: Works with any sized matrices that have dimensions
//           bigger than or equal to BLOCK_SIZE x BLOCK_SIZE. May
//           be slower than kernel 1 as a result of this flexibility.
#define MATRIX_KERNEL 1

//@@ If defined, forces all matrix dimensions to be a multiple of 16
//@@ This is required for Kernel 2 to work successfully
#define MATRIX_FORCE_TO_MULTIPLE_OF_16

//@@ Matrix dimensions are randomly generated between these two values
#define MATRIX_DIMENSION_MAX 1024
#define MATRIX_DIMENSION_MIN 16

//@@ Set to 1 to perform a single test for validating correctness of functions
//@@ Set to >1 to perform a repeat test for comparing speed of GPU to CPU
#define TEST_REPEAT_NUM 1

//@@ If defined, outputs results to debug.txt instead of to console
//#define DEBUG_OUTPUT_RESULTS

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);


int main(int argc, char** argv) {
	Matrix  M;
	Matrix  N;
	Matrix  P;
	int errorM = 0, errorN = 0;

	// Random seed
	srand(63);
	
	if(argc != 5 && argc != 4) {
		// Allocate and initialize the matrices
		int rmax = MATRIX_DIMENSION_MAX;
		int rmin = MATRIX_DIMENSION_MIN;

		// Generate random dimensions
		int a = max(rand() % rmax, rmin);
		int b = max(rand() % rmax, rmin);
		int c = max(rand() % rmax, rmin);

		#ifdef MATRIX_FORCE_TO_MULTIPLE_OF_16
		a = max(a / 16, 1) * 16;
		b = max(b / 16, 1) * 16;
		c = max(c / 16, 1) * 16;
		printf("Forced matrix dimensions to multiples of 16.\n");
		# endif

		// Generate matrices
		M  = AllocateMatrix(a, b, 1);
		N  = AllocateMatrix(M.width, c, 1);
		P  = AllocateMatrix(M.height, N.width, 0);

		printf("Chosen matrix dimensions:\n");
		printf("[%d,%d] * [%d,%d] = [%d,%d]\n", M.height, M.width, N.height, N.width, P.height, P.width);
	} else {
		// Allocate and read in matrices from disk
		int* params = NULL; //(int*)malloc(3 * sizeof(int));
	    unsigned int data_read = 3;
	    cutReadFilei(argv[1], &params, &data_read, true);
		if (data_read != 3) {
			printf("Error reading parameter file\n");
			return 1;
		}

		M  = AllocateMatrix(params[0], params[1], 0);
		N  = AllocateMatrix(params[1], params[2], 0);		
		P  = AllocateMatrix(params[0], params[2], 0);
		errorM = ReadFile(&M, argv[2]);
		errorN = ReadFile(&N, argv[3]);
		if(errorM  || errorN ) {
			printf("Error reading input files %d, %d\n", errorM, errorN);
			return 1;
		}
	}

	// Define timers
	float elapsedCPU = 0, elapsedGPU = 0;
	cudaEvent_t timerCPU1, timerCPU2, timerGPU1, timerGPU2;
	cudaEventCreate(&timerCPU1);
	cudaEventCreate(&timerCPU2);
	cudaEventCreate(&timerGPU1);
	cudaEventCreate(&timerGPU2);

	if (TEST_REPEAT_NUM <= 1) {
		////// Perform only 1 test //////

		// Compute M * N = P on the GPU
		cudaEventRecord(timerGPU1, 0);
		MatrixMulOnDevice(M, N, P);
		cudaEventRecord(timerGPU2, 0);

		cudaEventSynchronize(timerGPU1);
		cudaEventSynchronize(timerGPU2);
		cudaEventElapsedTime(&elapsedGPU, timerGPU1, timerGPU2);

		printf("GPU computation complete\n");

		// Compute M * N = P on the CPU
		Matrix reference = AllocateMatrix(P.height, P.width, 0);
		cudaEventRecord(timerCPU1, 0);
		computeGold(reference.elements, M.elements, N.elements, M.height, M.width, N.width);
		cudaEventRecord(timerCPU2, 0);

		cudaEventSynchronize(timerCPU1);
		cudaEventSynchronize(timerCPU2);
		cudaEventElapsedTime(&elapsedCPU, timerCPU1, timerCPU2);

		printf("CPU computation complete\n");

		// Output first 1000 elements for debugging
#ifdef DEBUG_OUTPUT_RESULTS
		int output_limit = P.height * P.width;
		FILE *fp;
		fp = fopen("debug.txt", "w");
#else
		int output_limit = 1000;
#endif
		printf("[  ID]       CPU : GPU\n");
		for (int k = 0; k < min(output_limit, P.height * P.width); k++) {
			float ecpu = reference.elements[k];
			float egpu = P.elements[k];
#ifdef DEBUG_OUTPUT_RESULTS
			fprintf(fp, "[%4d] %9f : %9f", k, ecpu, egpu);
			if (egpu < ecpu + 0.001f && egpu > ecpu - 0.001f)
				fprintf(fp, "\n");
			else
				fprintf(fp, " <--- DO NOT MATCH!\n");
#else
				printf("[%4d] %9f : %9f", k, ecpu, egpu);
				if (egpu < ecpu + 0.001f && egpu > ecpu - 0.001f)
					printf("\n");
				else
					printf(" <--- DO NOT MATCH!\n");
#endif
		}
#ifdef DEBUG_OUTPUT_RESULTS
		fclose(fp);
#endif

		// in this case check if the result is equivalent to the expected solution
		CUTBoolean res = cutComparefe(reference.elements, P.elements,
										P.height*P.width, 0.001f);
		printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

		printf("GPU took %fms\n", elapsedGPU);
		printf("CPU took %fms\n", elapsedCPU);

		// Output results to file
		WriteFile(P, "mm_output.txt");

		if(argc == 5) {
			WriteFile(P, argv[4]);
		} else if(argc == 2) {
			WriteFile(P, argv[1]);
		}
	} else {
		////// Perform multiple tests //////

		float totalElapsedGPU = 0;
		printf("---=== Begin matrixmul testing ===---\n");
		printf("> Run 1 CPU test\n");
		printf("Running CPU test...");
		Matrix reference = AllocateMatrix(P.height, P.width, 0);
		cudaEventRecord(timerCPU1, 0);
		computeGold(reference.elements, M.elements, N.elements, M.height, M.width, N.width);
		cudaEventRecord(timerCPU2, 0);
		cudaEventSynchronize(timerCPU1);
		cudaEventSynchronize(timerCPU2);
		cudaEventElapsedTime(&elapsedCPU, timerCPU1, timerCPU2);
		printf(" done! (%fms)\n", elapsedCPU);
		printf("> Run %d GPU tests\n", TEST_REPEAT_NUM);
		for (int testNumber = 0; testNumber < TEST_REPEAT_NUM; testNumber++) {
			printf("[%3d/%3d] Running GPU test...", testNumber + 1, TEST_REPEAT_NUM);
			cudaEventRecord(timerGPU1, 0);
			MatrixMulOnDevice(M, N, P);
			cudaEventRecord(timerGPU2, 0);
			cudaEventSynchronize(timerGPU1);
			cudaEventSynchronize(timerGPU2);
			cudaEventElapsedTime(&elapsedGPU, timerGPU1, timerGPU2);
			totalElapsedGPU += elapsedGPU;
			printf(" done! (%fms)\n", elapsedGPU);
		}
		printf("> Test complete!\n");
		float averageGPU = totalElapsedGPU / (TEST_REPEAT_NUM + 1);
		printf("CPU took %fms\n", elapsedCPU);
		printf("GPU took %fms (average)\n", averageGPU);
		int percentFaster = (elapsedCPU - averageGPU) / averageGPU * 100;
		if (percentFaster < 0)
			printf("GPU was %d%% slower\n", -percentFaster);
		else
			printf("GPU was %d%% faster\n", percentFaster);
		printf("---=== End matrixmul testing ===---\n");
	}

	// Free matrices
    FreeMatrix(&M);
    FreeMatrix(&N);
    FreeMatrix(&P);

    cudaDeviceReset();

	return 0;
}

void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P) {
    // Load M and N to the device
    Matrix Md = AllocateDeviceMatrix(M);
    CopyToDeviceMatrix(Md, M);
    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    // Allocate P on the device
    Matrix Pd = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Pd, P); // Clear memory

    // Set up kernel and launch
    int blockSize = 16;
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((Pd.width - 1) / dimBlock.x + 1, (Pd.height - 1) / dimBlock.y + 1);

#if MATRIX_KERNEL == 1
    MatrixMulKernel_BlockSize<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
#elif MATRIX_KERNEL == 2
    MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
#else
    printf("Invalid kernel number selected: %d\n", MATRIX_KERNEL);
#endif

    // Read P from the device
    CopyFromDeviceMatrix(P, Pd); 

    // Free device matrices
    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M) {
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init) {
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    // don't allocate memory on option 2
    if(init == 2)
		return M;
		
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++) {
		M.elements[i] = (init == 0) ? (0.0f) : (rand()*3 / (float)RAND_MAX);
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost) {
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice) {
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
					cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M) {
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M) {
    free(M->elements);
    M->elements = NULL;
}

// Read a floating point matrix in from file
// Returns zero if the number of elements read is 
//  equals M.height * M.width, and 1 otherwise
int ReadFile(Matrix* M, char* file_name) {
	unsigned int data_read = M->height*M->width;
	cutReadFilef(file_name, &(M->elements), &data_read, true);
	return (data_read != (M->height * M->width));
}

// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name) {
    cutWriteFilef(file_name, M.elements, M.width*M.height,
                       0.0001f);
}
