#include "matrixmul_kernel.h"

#define BLOCK_SIZE 16

__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int i;

	float PdValue = 0.0f;

	if (row < P.height && col < P.width) {
		for (i = 0; i < M.width; ++i) {
			PdValue += M.elements[row * M.width + i] * N.elements[i * N.width + col];
		}
	P.elements[row * P.width + col] = PdValue;
	}
}

/*
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

	int rowM = ty + by * BLOCK_SIZE;
	int colN = tx + bx * BLOCK_SIZE;

	int mLimit = (wM - 1) / BLOCK_SIZE + 1 - 1;
	if (blockIdx.y * blockDim.y + threadIdx.y < P.height && blockIdx.x * blockDim.x + threadIdx.x < P.width) {
		for (unsigned int m = 0; m < (wM - 1) / BLOCK_SIZE + 1; m++) {
			__shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
			__shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];

			int colM = m * BLOCK_SIZE + tx;
			int rowN = m * BLOCK_SIZE + ty;

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
	}

	int p = nBegin + nStep * by;
	P.elements[p + wN * ty + tx] = Psub;
}
*/
