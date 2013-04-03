################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../matrixmul_gold.cpp 

CU_SRCS += \
../cutilmk2.cu \
../matrixmul.cu \
../matrixmul_kernel.cu 

CU_DEPS += \
./cutilmk2.d \
./matrixmul.d \
./matrixmul_kernel.d 

OBJS += \
./cutilmk2.o \
./matrixmul.o \
./matrixmul_gold.o \
./matrixmul_kernel.o 

CPP_DEPS += \
./matrixmul_gold.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/Developer/NVIDIA/CUDA-5.0/samples/common/inc -G -g -O0 -gencode arch=compute_11,code=sm_11 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -I/Developer/NVIDIA/CUDA-5.0/samples/common/inc -O0 -g -gencode arch=compute_11,code=compute_11 -gencode arch=compute_11,code=sm_11  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/Developer/NVIDIA/CUDA-5.0/samples/common/inc -G -g -O0 -gencode arch=compute_11,code=sm_11 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I/Developer/NVIDIA/CUDA-5.0/samples/common/inc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


