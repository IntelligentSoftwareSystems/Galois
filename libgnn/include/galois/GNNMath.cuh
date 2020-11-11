#ifndef GALOIS_GNN_MATH_CUDA
#define GALOIS_GNN_MATH_CUDA
#include "galois/GNNTypes.h"
#include "galois/CUDAUtil.h"

namespace galois {

extern bool cublas_is_init;
extern cublasHandle_t global_cublas_handle;

//! Initializes the cublas handle to use cublas on GPUs.
void InitCuBLAS();

//! Takes 2 *row-major* matrices and does a matrix multiply on the GPU using
//! CuBLAS.
void CBlasSGEMMGPU(const cublasOperation_t trans_a,
                   const cublasOperation_t trans_b, size_t input_rows,
                   size_t input_columns, size_t output_columns,
                   const GNNFloat* a, const GNNFloat* b, GNNFloat* output);

} // namespace galois
#endif
