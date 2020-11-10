#ifdef GALOIS_ENABLE_GPU
//! @file CUDAUtil.h
//! Contains various utility functions for CUDA.
#pragma once
#include <cuda.h>
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
      GALOIS_LOG_ERROR("CUDA error: {}", cudaGetErrorString(error));           \
      exit(EXIT_FAILURE);                                                      \
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

#endif
