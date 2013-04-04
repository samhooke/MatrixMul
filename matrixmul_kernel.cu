#include "matrixmul_kernel.h"

#define BLOCK_SIZE 16

__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P) {
	int wM = M.width;
	int wN = N.width;

	const unsigned int bx = blockIdx.x;
	const unsigned int by = blockIdx.y;
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;

	const unsigned int mBegin = wM * (by * BLOCK_SIZE);
	const unsigned int mEnd = mBegin + wM;
	//const unsigned int mEnd = mBegin + wM - 1;
	const unsigned int mStep = BLOCK_SIZE;

	const unsigned int nBegin = BLOCK_SIZE * bx;
	const unsigned int nStep = BLOCK_SIZE * wN;

	float Psub = 0;

	unsigned int m, n;

	for (m = mBegin, n = nBegin; m < mEnd - BLOCK_SIZE; m += mStep, n += nStep) {
	//for (m = mBegin, n = nBegin; m <= mEnd; m += mStep, n += nStep) {
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

	__shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];

	if((ty + by * BLOCK_SIZE < wM) && ((m - mBegin) + tx < wN))
		Ms[ty][tx] = M.elements[m + wM * ty + tx];
	else
		Ms[ty][tx] = 0;
	if((ty + (m - mBegin) < wM) && (BLOCK_SIZE * bx + tx < wN))
		Ns[ty][tx] = N.elements[n + wN * ty + tx];
	else
		Ns[ty][tx] = 0;

	__syncthreads();

	for (int k = 0; k < BLOCK_SIZE; k++) {
		Psub += Ms[ty][k] * Ns[k][tx];
	}
	__syncthreads();

    if(by * BLOCK_SIZE + ty < wM && bx * BLOCK_SIZE + tx < wN) {
        //int p = (by * BLOCK_SIZE + ty) * wM + (bx * BLOCK_SIZE + tx);
    	//P.elements[p] = Psub;

        int c = wN * BLOCK_SIZE * by + BLOCK_SIZE * bx;
        P.elements[c + wN * ty + tx] = Psub;
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

	//float Psub = 0;
	float Psub = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	for (unsigned int m = 0; m < (wM - 1) / BLOCK_SIZE + 1; m++) {
		__shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];

		int pM = mBegin + (m * mStep); // Block start in M
		int pN = nBegin + (m * nStep); // Block start in N

		int gM = pM + wM * ty + tx; // Cell number
		int gN = pN + wN * ty + tx; // Cell number

		//if (gM > 271) {
		//	Ms[ty][tx] = 0;
		//	Ns[ty][tx] = 0;
		//} else {
		//	Ms[ty][tx] = M.elements[gM];
		//	Ns[ty][tx] = N.elements[gN];
		//}

		Ms[ty][tx] = M.elements[gM];
		Ns[ty][tx] = N.elements[gN];

		__syncthreads();



		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Psub[k] += Ms[ty][k] * Ns[k][tx];
			//Psub += Ms[ty][k] * Ns[k][tx];
			//Psub = gM;
		}
		__syncthreads();
	}

	int p = nBegin + nStep * by;
	P.elements[p + wN * ty + tx] = Psub;
}
*/
/*
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
*/
/*
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int i;

	float PdValue = 0.0f;

	int mBegin = M.width * BLOCK_SIZE * blockIdx.y;
	int mStep = BLOCK_SIZE;

	int nBegin = blockIdx.x * BLOCK_SIZE;
	int nStep = N.width * BLOCK_SIZE;

	if (row < P.height && col < P.width) {
		for (unsigned int m = 0; m < (M.width - 1) / BLOCK_SIZE + 1; m++) {
			__shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
			__shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];

			int colM = m * BLOCK_SIZE + threadIdx.x;
			int rowN = m * BLOCK_SIZE + threadIdx.y;

			int pM = mBegin + (m * mStep);
			int pN = nBegin + (m * nStep);

			int gM = pM + M.width * threadIdx.y + threadIdx.x;
			int gN = pN + N.width * threadIdx.y + threadIdx.x;

			Ms[threadIdx.y][threadIdx.x] = M.elements[gM];
			Ns[threadIdx.y][threadIdx.x] = N.elements[gN];

			__syncthreads();

			for (int k = 0; k < BLOCK_SIZE; ++k) {
				PdValue += Ms[threadIdx.y][k] * Ns[k][threadIdx.x];
			}

			__syncthreads();
		}
		P.elements[row * P.width + col] = PdValue;
	}
}
*/
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
