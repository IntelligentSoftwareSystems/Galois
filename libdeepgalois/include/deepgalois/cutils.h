#pragma once
#include <cuda.h>
#include <curand.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <iostream>

// CUDA: use 256 threads per block
const int CUDA_NUM_THREADS = 256;

// CUDA: number of blocks for threads.
inline int CUDA_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

inline unsigned CudaTest(const char* msg) {
  cudaError_t e;
  // cudaThreadSynchronize();
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(-1);
  }
  return 0;
}

inline const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  default:
    break;
  }
  return "Unknown cublas status";
}

inline const char* cusparseGetErrorString(cusparseStatus_t error) {
  switch (error) {
  case CUSPARSE_STATUS_SUCCESS:
    return "CUSPARSE_STATUS_SUCCESS";
  case CUSPARSE_STATUS_NOT_INITIALIZED:
    return "CUSPARSE_STATUS_NOT_INITIALIZED";
  case CUSPARSE_STATUS_ALLOC_FAILED:
    return "CUSPARSE_STATUS_ALLOC_FAILED";
  case CUSPARSE_STATUS_INVALID_VALUE:
    return "CUSPARSE_STATUS_INVALID_VALUE";
  case CUSPARSE_STATUS_ARCH_MISMATCH:
    return "CUSPARSE_STATUS_ARCH_MISMATCH";
  case CUSPARSE_STATUS_MAPPING_ERROR:
    return "CUSPARSE_STATUS_MAPPING_ERROR";
  case CUSPARSE_STATUS_EXECUTION_FAILED:
    return "CUSPARSE_STATUS_EXECUTION_FAILED";
  case CUSPARSE_STATUS_INTERNAL_ERROR:
    return "CUSPARSE_STATUS_INTERNAL_ERROR";
  case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  case CUSPARSE_STATUS_ZERO_PIVOT:
    return "CUSPARSE_STATUS_ZERO_PIVOT";
  default:
    break;
  }
  return "Unknown cusparse status";
}

inline const char* curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  default:
    break;
  }
  return "Unknown curand status";
}

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition)                                                  \
  do {                                                                         \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "error %d: Cuda error in file '%s' in line %i : %s.\n",  \
              error, __FILE__, __LINE__, cudaGetErrorString(error));           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(condition)                                                \
  do {                                                                         \
    cublasStatus_t status = condition;                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr,                                                          \
              "error %d: cuBLAS error in file '%s' in line %i : %s.\n",        \
              status, __FILE__, __LINE__, cublasGetErrorString(status));       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUSPARSE_CHECK(condition)                                              \
  do {                                                                         \
    cusparseStatus_t status = condition;                                       \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      fprintf(stderr,                                                          \
              "error %d: cuSPARSE error in file '%s' in line %i : %s.\n",      \
              status, __FILE__, __LINE__, cusparseGetErrorString(status));     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CURAND_CHECK(condition)                                                \
  do {                                                                         \
    curandStatus_t status = condition;                                         \
    if (status != CURAND_STATUS_SUCCESS) {                                     \
      fprintf(stderr,                                                          \
              "error %d: cuBLAS error in file '%s' in line %i : %s.\n",        \
              status, __FILE__, __LINE__, curandGetErrorString(status));       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

inline void print_device_vector(size_t n, const float_t* d_x,
                                std::string name = "x") {
  float_t* h_x = new float_t[n];
  CUDA_CHECK(cudaMemcpy(h_x, d_x, n * sizeof(float_t), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; i++)
    std::cout << name << "[" << i << "]=" << h_x[i] << "\n";
  delete[] h_x;
}

inline void print_device_int_vector(size_t n, const int* d_x,
                                    std::string name = "x") {
  int* h_x = new int[n];
  CUDA_CHECK(cudaMemcpy(h_x, d_x, n * sizeof(int), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; i++)
    std::cout << name << "[" << i << "]=" << h_x[i] << "\n";
  delete[] h_x;
}
