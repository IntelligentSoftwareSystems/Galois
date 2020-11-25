#ifndef GALOIS_GNN_MATH_CUDA
#define GALOIS_GNN_MATH_CUDA
#include "galois/GNNTypes.h"
#include "galois/CUDAUtil.h"

namespace galois {

extern bool cublas_is_init;
extern cublasHandle_t global_cublas_handle;
extern bool curand_is_init;
extern curandGenerator_t global_curand_generator;

//! Initializes the cublas handle to use cublas on GPUs.
void InitCuBLAS();
//! Initializes the curand RNG
void InitCuRAND();

//! Initializes an array with random numbers (0.0, 1.0]
void CuRANDUniformRNG(GNNFloat* array_to_fill, size_t num_elements);

//! Takes 2 *row-major* matrices and does a matrix multiply on the GPU using
//! CuBLAS.
void CBlasSGEMMGPU(const cublasOperation_t trans_a,
                   const cublasOperation_t trans_b, size_t input_rows,
                   size_t input_columns, size_t output_columns,
                   const GNNFloat* a, const GNNFloat* b, GNNFloat* output);

//! Runs softmax + cross entropy on masked nodes. Will not overwrite all of
//! the output, so make sure it's been zero'd out beforehand.
//! At this point in time cross entropy is ignored because it only calculates a
//! loss value which doesn't really do anything for us at the moment.
__global__ void
SoftmaxCrossEntropyForward(char* mask, size_t num_nodes, size_t feature_length,
                           const galois::GNNFloat* input_embeddings,
                           galois::GNNFloat* output);

//! Derivative of cross entropy (to get error of prediction) then derivavtive
//! of the softmax.
__global__ void
SoftmaxCrossEntropyBackward(char* mask, size_t num_nodes, size_t feature_length,
                            const galois::GNNFloat* predictions,
                            const galois::GNNLabel* ground_truth,
                            galois::GNNFloat* output_gradient);

//! Given a vector, apply a softmax on some specified # of elements and save
//! the result to the specified output. Since this is a device function,
//! all pointers should be to GPU memory.
__device__ void DoSoftmax(size_t vector_length, const GNNFloat* input,
                          GNNFloat* output);

} // namespace galois
#endif
