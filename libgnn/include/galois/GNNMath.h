#pragma once

#include "galois/GNNTypes.h"
#include <mkl.h>

namespace galois {

//! Find max index in a vector of some length
size_t MaxIndex(const size_t length, const GNNFloat* vector);
//! Given 2 float array pointers, do element wise addition of length elements
//! Can be called in parallel sections as its sigle threaded code
void VectorAdd(size_t length, const GNNFloat* a, const GNNFloat* b,
               GNNFloat* output);

//! Does a softmax operation on the input vector and saves result to output
//! vector; single threaded so it can be called in a parallel section
void GNNSoftmax(const size_t vector_length, const GNNFloat* input,
                GNNFloat* output);
//! Get derivative of softmax given the forward pass's input, the derivative
//! from loss calculation, and a temp vector to store intermediate results.
//! Everything is the same size.
void GNNSoftmaxDerivative(const size_t vector_length,
                          const GNNFloat* prev_output,
                          const GNNFloat* prev_output_derivative,
                          GNNFloat* temp_vector, GNNFloat* output);
//! Performs cross entropy given a ground truth and input and returns the loss
//! value.
galois::GNNFloat GNNCrossEntropy(const size_t vector_length,
                                 const GNNFloat* ground_truth,
                                 const GNNFloat* input);
//! Derivative of cross entropy; gradients saved into an output vector.
void GNNCrossEntropyDerivative(const size_t vector_length,
                               const GNNFloat* ground_truth,
                               const GNNFloat* input, GNNFloat* gradients);
//! Calls into a library BLAS call to do matrix muliply; uses default alpha/beta
void CBlasSGEMM(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                size_t input_rows, size_t input_columns, size_t output_columns,
                const GNNFloat* a, const GNNFloat* b, GNNFloat* output);

} // namespace galois
