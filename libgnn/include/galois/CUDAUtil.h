#ifndef GALOIS_CUDA_UTIL
#define GALOIS_CUDA_UTIL
//! @file CUDAUtil.h
//! Contains various utility functions for CUDA.
#include <cuda.h>
#include <cublas_v2.h>
#include "galois/Logging.h"

// TODO check these
#define CHUNK_SIZE 256
#define TB_SIZE 256
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_NUM_CLASSES 128
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

#define CUDA_CHECK(condition)                                                  \
  do {                                                                         \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      GALOIS_LOG_FATAL("CUDA error: {}", cudaGetErrorString(error));           \
    }                                                                          \
  } while (0)

#define CUDA_FREE(ptr)                                                         \
  do {                                                                         \
    if (ptr) {                                                                 \
      CUDA_CHECK(cudaFree(ptr));                                               \
      ptr = nullptr;                                                           \
    }                                                                          \
  } while (0)

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

#define CUBLAS_CHECK(condition)                                                \
  do {                                                                         \
    cublasStatus_t status = condition;                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      GALOIS_LOG_ERROR("CuBLAS error code : {}", status);                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#endif
