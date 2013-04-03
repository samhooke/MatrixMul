#include "matrixmul_kernel.h"

#define BLOCK_SIZE 16

__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P) {
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

	for (unsigned int m = 0; m < wM / BLOCK_SIZE; m++) {
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
