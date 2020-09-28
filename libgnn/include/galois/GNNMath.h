#pragma once

#include "galois/GNNTypes.h"
#include <cblas.h>

namespace galois {

//! Given 2 float array pointers, do element wise addition of length elements
//! Can be called in parallel sections as its sigle threaded code
void VectorAdd(size_t length, const GNNFloat* a, const GNNFloat* b,
               GNNFloat* output);

//! Calls into a library BLAS call to do matrix muliply; uses default alpha/beta
void CBlasSGEMM(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                size_t input_rows, size_t input_columns, size_t output_columns,
                const GNNFloat* a, const GNNFloat* b, GNNFloat* output);

} // namespace galois
