# MatrixMul

A task done for the unit "GPU Architecture and Programming (ENG722S2)". Implements tiled matrix multiplication in CUDA, through two methods.

* Kernel 1: Matrix dimensions must be multiples of BLOCK_SIZE
* Kernel 2: Matrix dimensions can be arbitrary (at the cost of a slight drop in performance)

# cutil.h

This task was supposed to use _cutil.h_, however support for that has been dropped in CUDA 5.x. Included in the CUDA SDK is _helper_functions.h_, which is meant to replace the functionality of the deprecated _cutil.h_. As a result, cutilmk2.h was created which replicates some of the missing functions by calling helper_functions.h.

# NVIDIA

As was part of the assignment, much of the original source was based upon code samples from NVIDIA. In particular:

* matrixmul.cu
* matrixmul.h
* matrixmul_gold.cpp

Though matrixmul.cu was modified substantially to include the following functionality:

* Timing metrics
* Multiple kernel invocations
* Kernel selection
* Matrix generation parameters
