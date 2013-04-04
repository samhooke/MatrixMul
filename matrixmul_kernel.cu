#include "matrixmul_kernel.h"

#define BLOCK_SIZE 16

// Works with any sized matrices whose dimensions are BLOCK_SIZE or bigger
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P) {
	const unsigned int wM = M.width;
	const unsigned int wN = N.width;

	const unsigned int bx = blockIdx.x;
	const unsigned int by = blockIdx.y;
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;

	const unsigned int mBegin = wM * (by * BLOCK_SIZE);
	const unsigned int mEnd = mBegin + wM;
	const unsigned int mStep = BLOCK_SIZE;

	const unsigned int nBegin = BLOCK_SIZE * bx;
	const unsigned int nStep = BLOCK_SIZE * wN;

	float Psub = 0;

	unsigned int m, n;

	// Perform all non-edge case multiplications
	for (m = mBegin, n = nBegin; m < mEnd - BLOCK_SIZE; m += mStep, n += nStep) {
		__shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];

		Ms[ty][tx] = M.elements[m + wM * ty + tx];
		Ns[ty][tx] = N.elements[n + wN * ty + tx];

		__syncthreads();

		for (unsigned int k = 0; k < BLOCK_SIZE; k++) {
			Psub += Ms[ty][k] * Ns[k][tx];
		}

		__syncthreads();
	}

	// Perform edge case multiplications
	// Some logic inferred from: https://github.com/yqzhang/Parallel-Computation/
	__shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];

	if ((m - mBegin) + tx < wM)
		Ms[ty][tx] = M.elements[m + wM * ty + tx];
	else
		Ms[ty][tx] = 0;
	if (ty + (m - mBegin) < wM)
		Ns[ty][tx] = N.elements[n + wN * ty + tx];
	else
		Ns[ty][tx] = 0;

	__syncthreads();

	for (unsigned int k = 0; k < BLOCK_SIZE; k++) {
		Psub += Ms[ty][k] * Ns[k][tx];
	}
	__syncthreads();

	if ((bx * BLOCK_SIZE + tx) < wN) {
        int p = wN * BLOCK_SIZE * by + BLOCK_SIZE * bx;
        P.elements[p + wN * ty + tx] = Psub;
    }
}

// Only works with matrices whose dimensions are multiples of BLOCK_SIZE
__global__ void MatrixMulKernel_BlockSize(Matrix M, Matrix N, Matrix P) {
	const unsigned int wM = M.width;
	const unsigned int wN = N.width;

	const unsigned int bx = blockIdx.x;
	const unsigned int by = blockIdx.y;
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;

	const unsigned int mBegin = wM * BLOCK_SIZE * by;
	const unsigned int mStep = BLOCK_SIZE;

	const unsigned int nBegin = bx * BLOCK_SIZE;
	const unsigned int nStep = wN * BLOCK_SIZE;

	float Psub = 0;

	for (unsigned int m = 0; m < (wM - 1) / BLOCK_SIZE + 1; m++) {
		__shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];

		Ms[ty][tx] = M.elements[(mBegin + (m * mStep)) + wM * ty + tx];
		Ns[ty][tx] = N.elements[(nBegin + (m * nStep)) + wN * ty + tx];

		__syncthreads();

		for (unsigned int k = 0; k < BLOCK_SIZE; ++k) {
			Psub += Ms[ty][k] * Ns[k][tx];
		}
		__syncthreads();
	}

	int p = nBegin + nStep * by;
	P.elements[p + wN * ty + tx] = Psub;
}
