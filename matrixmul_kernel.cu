#include "matrixmul_kernel.h"

#define BLOCK_SIZE 16

// Works with any sized matrices whose dimensions are BLOCK_SIZE or bigger
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P) {
	int wM = M.width;
	int wN = N.width;

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

		for (int k = 0; k < BLOCK_SIZE; k++) {
			Psub += Ms[ty][k] * Ns[k][tx];
		}

		__syncthreads();
	}

	// Perform edge case multiplications
	__shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];

	bool c1 = ((m - mBegin) + tx < wM);
	bool c2 = (ty + (m - mBegin) < wM);
	bool c3 = (bx * BLOCK_SIZE + tx) < wN;

	if (c1)
		Ms[ty][tx] = M.elements[m + wM * ty + tx];
	else
		Ms[ty][tx] = 0;
	if (c2)
		Ns[ty][tx] = N.elements[n + wN * ty + tx];
	else
		Ns[ty][tx] = 0;

	__syncthreads();

	for (int k = 0; k < BLOCK_SIZE; k++) {
		Psub += Ms[ty][k] * Ns[k][tx];
	}
	__syncthreads();

	if (c3) {
        int p = wN * BLOCK_SIZE * by + BLOCK_SIZE * bx;
        P.elements[p + wN * ty + tx] = Psub;
    }
}

// Only works with matrices whose dimensions are multiples of BLOCK_SIZE
__global__ void MatrixMulKernel_BlockSize(Matrix M, Matrix N, Matrix P) {
	int wM = M.width;
	int wN = N.width;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int mBegin = wM * BLOCK_SIZE * by;
	int mStep = BLOCK_SIZE;

	int nBegin = bx * BLOCK_SIZE;
	int nStep = wN * BLOCK_SIZE;

	float Psub = 0;

	for (unsigned int m = 0; m < (wM - 1) / BLOCK_SIZE + 1; m++) {
		__shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];

		int pM = mBegin + (m * mStep);
		int pN = nBegin + (m * nStep);

		int gM = pM + wM * ty + tx;
		int gN = pN + wN * ty + tx;

		Ms[ty][tx] = M.elements[gM];
		Ns[ty][tx] = N.elements[gN];

		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Psub += Ms[ty][k] * Ns[k][tx];
		}
		__syncthreads();
	}

	int p = nBegin + nStep * by;
	P.elements[p + wN * ty + tx] = Psub;
}
