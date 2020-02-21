#pragma once
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <driver_types.h>

// CUDA: use 256 threads per block
const int CUDA_NUM_THREADS = 256;

// CUDA: number of blocks for threads.
inline int CUDA_GET_BLOCKS(const int N) {
	return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) {    \
      fprintf(stderr, "error %d: Cuda error in file '%s' in line %i : %s.\n", \
      error, __FILE__, __LINE__, cudaGetErrorString(error) );                    \
      exit(EXIT_FAILURE);                                                     \
    } \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition;   \
    if (status != CUBLAS_STATUS_SUCCESS) \
      ;      \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

