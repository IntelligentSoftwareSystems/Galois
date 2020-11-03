#ifdef GALOIS_ENABLE_GPU
//! @file CUDAUtil.h
//! Contains various utility functions for CUDA.
#pragma once
#include <cuda.h>
#include "galois/Logging.h"

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

#endif
