#ifndef GALOIS_CUDA_UTIL
#define GALOIS_CUDA_UTIL
//! @file CUDAUtil.h
//! Contains various utility functions for CUDA.
//! Taken and revised+added to from here
//! https://github.com/BVLC/caffe/blob/master/include/caffe/util/device_alternate.hpp
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include "galois/Logging.h"

// TODO check these too and make sure they make sense
// CUDA: use 256 threads per block
const int CUDA_NUM_THREADS = 256;

// CUDA: number of blocks for threads.
inline int CUDA_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// TODO check these
#define CHUNK_SIZE 256
#define TB_SIZE 256
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_NUM_CLASSES 128
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

//! Wrap a CUDA call with this to auto-check if it returns any error
#define CUDA_CHECK(condition)                                                  \
  do {                                                                         \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      GALOIS_LOG_FATAL("CUDA error: {}", cudaGetErrorString(error));           \
    }                                                                          \
  } while (0)

//! Frees a pointer allocated by cuda malloc
#define CUDA_FREE(ptr)                                                         \
  do {                                                                         \
    if (ptr) {                                                                 \
      CUDA_CHECK(cudaFree(ptr));                                               \
      ptr = nullptr;                                                           \
    }                                                                          \
  } while (0)

//! Call this after a cuda call to make sure it set any error flags
#define CUDA_TEST(msg)                                                         \
  do {                                                                         \
    cudaError_t e;                                                             \
    cudaDeviceSynchronize();                                                   \
    if (cudaSuccess != (e = cudaGetLastError())) {                             \
      GALOIS_LOG_ERROR("{}: {}", msg, e);                                      \
      GALOIS_LOG_ERROR("{}", cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

//! Basic kernel loop for CUDA threads
//! Caffe describes it as "grid stride"
#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

//! Wrap a CuBLAS call with this to check if it threw any errors
#define CUBLAS_CHECK(condition)                                                \
  do {                                                                         \
    cublasStatus_t status = condition;                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      GALOIS_LOG_ERROR("CuBLAS error code : {}", status);                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

//! Wrap a CuRAND call with this to check if it threw any errors
#define CURAND_CHECK(condition)                                                \
  do {                                                                         \
    curandStatus_t status = condition;                                         \
    if (status != CURAND_STATUS_SUCCESS) {                                     \
      GALOIS_LOG_ERROR("CuRAND error code : {}", status);                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#endif
