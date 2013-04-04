#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P);
__global__ void MatrixMulKernel_BlockSize(Matrix M, Matrix N, Matrix P);

#endif // #ifndef _MATRIXMUL_KERNEL_H_
