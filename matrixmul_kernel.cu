#include "matrixmul_kernel.h"

#define BLOCK_SIZE 16

__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P) {
	int wM = M.width;
	int wN = N.width;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int mBegin = wM * (by * BLOCK_SIZE);
	int mEnd = mBegin + wM;
	int mStep = BLOCK_SIZE;

	int nBegin = BLOCK_SIZE * bx;
	int nStep = BLOCK_SIZE * wN;

	float Psub = 0;

	//for (unsigned int m = 0; m < (wM - 1) / BLOCK_SIZE + 1; m++) {
	unsigned int m, n;
	for (m = mBegin, n = nBegin; m < mEnd - BLOCK_SIZE; m += mStep, n += nStep) {
		__shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];

		//int pM = mBegin + (m * mStep); // Block start in M
		//int pN = nBegin + (m * nStep); // Block start in N

		//int gM = pM + wM * ty + tx; // Cell number
		//int gN = pN + wN * ty + tx; // Cell number

		//if (gM > 271) {
		//	Ms[ty][tx] = 0;
		//	Ns[ty][tx] = 0;
		//} else {
		//	Ms[ty][tx] = M.elements[gM];
		//	Ns[ty][tx] = N.elements[gN];
		//}

		//Ms[ty][tx] = M.elements[gM];
		//Ns[ty][tx] = N.elements[gN];

		//Ms[ty][tx] = M.elements[gM];
		//Ns[ty][tx] = N.elements[gN];

		/*
        if((ty + by * BLOCK_SIZE < wM) && ((pM - mBegin) + tx < wM))
        	Ms[ty][tx] = M.elements[pM + wM * ty + tx];
        else
        	Ms[ty][tx] = 0;
        if((ty + (pN - nBegin) < wN) && (BLOCK_SIZE * bx + tx < wN))
        	Ns[ty][tx] = N.elements[pN + wN * ty + tx];
        else
        	Ns[ty][tx] = 0;
        */
        if((ty + by * BLOCK_SIZE < wM) && ((m - mBegin) + tx < wM))
        	Ms[ty][tx] = M.elements[m + wM * (ty) + tx];
        else
        	Ms[ty][tx] = 0;
        if((ty + (m - mBegin) < wN) && (BLOCK_SIZE * bx + tx < wN))
        	Ns[ty][tx] = N.elements[n + wN * (ty) + tx];
        else
        	Ns[ty][tx] = 0;

		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Psub += Ms[ty][k] * Ns[k][tx];
		}
		__syncthreads();
	}

	int p = nBegin + nStep * by;
	P.elements[p + wN * ty + tx] = Psub;
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
/*
//__global__ void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P) {
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P) {
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	// Identify the row and column of the Pd element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	int limit;

	float Pvalue = 0;
	// Loop over the Md and Nd tiles required to compute the Pd element
	for (int m = 0; m < (ceil((float)M.width/(float)TILE_WIDTH)); ++m) {
		//suppose M.width is 66 then the 1st four tiles are 16 elements wide .. and the last one is 2 .. here we calculate m which is used in calculation to indicate this
		(m == (ceil((float)M.width/(float)TILE_WIDTH)-1)) ? limit = (M.width - TILE_WIDTH*(floor((float)M.width/(float)TILE_WIDTH))): limit = TILE_WIDTH ;

		//calculate all tiles except the bottom and most right ones (this is because all tiles here are guaranteed to be of tile_width*tile_width size)
		if ((bx < (ceil((float)N.width/(float)TILE_WIDTH)-1)) && (by <(ceil((float)M.height/(float)TILE_WIDTH)-1))) {
			Mds[ty][tx] = M.elements[Row*M.width + (m*TILE_WIDTH + tx)];
			Nds[ty][tx] = N.elements[Col + (m*TILE_WIDTH + ty)*N.width];
			__syncthreads();
			for (int k = 0; k < limit; ++k) {
				Pvalue += Mds[ty][k] * Nds[k][tx];
			}
		}

		//calculate the bottom right most corner tile (it is not guaranteed to be tile_width in any direction .. can be less)
		else if ((bx == (ceil((float)N.width/(float)TILE_WIDTH)-1)) && (by == (ceil((float)M.height/(float)TILE_WIDTH)-1)) && ((N.width % TILE_WIDTH) != 0) && ((M. height % TILE_WIDTH) != 0)) {
			if ((tx < (N.width - TILE_WIDTH*(floor((float)N.width/(float)TILE_WIDTH)))) && (ty < (M.height - TILE_WIDTH*(floor((float)M.height/(float)TILE_WIDTH))))) {
				Mds[ty][tx] = M.elements[Row*M.width + (m*TILE_WIDTH + tx)];
				Nds[ty][tx] = N.elements[Col + (m*TILE_WIDTH + ty)*N.width];
				__syncthreads();
				for (int k = 0; k < limit; ++k) {
					Pvalue += Mds[ty][k] * Nds[k][tx];
				}
			}
		}

		//calculate the right most column except the corner tile .. here the width of the right most column is less than tile_width
		else if ((bx == (ceil((float)N.width/(float)TILE_WIDTH)-1)) && ((N.width % TILE_WIDTH) != 0) && (by != (ceil((float)M.height/(float)TILE_WIDTH)-1))) {
			if (tx < (N.width - TILE_WIDTH*(floor((float)N.width/(float)TILE_WIDTH)))) {
				Mds[ty][tx] = M.elements[Row*M.width + (m*TILE_WIDTH + tx)];
				Nds[ty][tx] = N.elements[Col + (m*TILE_WIDTH + ty)*N.width];
				__syncthreads();
				for (int k = 0; k < limit; ++k) {
					Pvalue += Mds[ty][k] * Nds[k][tx];
				}
			}
		}

		//calculate the bottom line tiles except the right most corner tile ,, this is the case where the height of the tile is less than tile_width
		else if ((by == (ceil((float)M.height/(float)TILE_WIDTH)-1)) && ((M. height % TILE_WIDTH) != 0) && (bx != (ceil((float)N.width/(float)TILE_WIDTH)-1))) {
			if (ty < (M.height - TILE_WIDTH*(floor((float)M.height/(float)TILE_WIDTH)))) {
				Mds[ty][tx] = M.elements[Row*M.width + (m*TILE_WIDTH + tx)];
				Nds[ty][tx] = N.elements[Col + (m*TILE_WIDTH + ty)*N.width];
				__syncthreads();
				for (int k = 0; k < limit; ++k) {
					Pvalue += Mds[ty][k] * Nds[k][tx];
				}
			}
		}
		__syncthreads();
	}
	P.elements[Row*P.width+Col] = Pvalue;
}
*/
