#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include "galois/GNNMath.h"
#include "galois/Logging.h"

size_t galois::MaxIndex(const size_t length, const GNNFloat* vector) {
  size_t index     = 0;
  GNNFloat cur_max = vector[0];

  for (size_t i = 1; i < length; i++) {
    if (vector[i] > cur_max) {
      index   = i;
      cur_max = vector[i];
    }
  }

  return index;
}

void galois::VectorAdd(size_t length, const GNNFloat* a, const GNNFloat* b,
                       GNNFloat* output) {
#ifdef __AVX2__
  constexpr size_t vectorization_length =
      8; // for 32-bit floating point in AVX2; TODO AVX512
  // can only do up to a particular multiple due to alignment
  const size_t aligned_end = length - length % vectorization_length;
  // do add via vector ops
  for (size_t i = 0; i < aligned_end; i += vectorization_length) {
    _mm256_storeu_ps(&output[i], _mm256_add_ps(_mm256_loadu_ps(&a[i]),
                                               _mm256_loadu_ps(&b[i])));
  }

  // handle the rest
  for (size_t i = aligned_end; i < length; ++i) {
    output[i] = a[i] + b[i];
  }
#else
  // no vector -> trivial loop add
  for (size_t i = 0; i < length; ++i) {
    output[i] = a[i] + b[i];
  }
#endif
}

void galois::GNNSoftmax(const size_t vector_length, const GNNFloat* input,
                        GNNFloat* output) {
  const GNNFloat max_element =
      *(std::max_element(input, input + vector_length));
  GNNFloat denominator = 0;
  // normalize all elements using exponentional of max element
  for (size_t i = 0; i < vector_length; i++) {
    output[i] = std::exp(input[i] - max_element);
    denominator += output[i];
  }
  // divide all by total to get a distribution
  for (size_t i = 0; i < vector_length; i++) {
    output[i] /= denominator;
  }
}

void galois::GNNSoftmaxDerivative(const size_t vector_length,
                                  const GNNFloat* prev_output,
                                  const GNNFloat* prev_output_derivative,
                                  GNNFloat* temp_vector, GNNFloat* output) {
  for (size_t i = 0; i < vector_length; i++) {
    for (size_t j = 0; j < vector_length; j++) {
      temp_vector[j] = (j == i) ? prev_output[i] * (1.0 - prev_output[i])
                                : -prev_output[j] * prev_output[i];
    }
    // TODO is sdot using threads? if so this is a nested parallelism problem
    output[i] =
        cblas_sdot(vector_length, prev_output_derivative, 1, temp_vector, 1);
  }
}

galois::GNNFloat galois::GNNCrossEntropy(const size_t vector_length,
                                         const GNNFloat* ground_truth,
                                         const GNNFloat* input) {
  GNNFloat loss = 0.0;

  for (size_t i = 0; i < vector_length; i++) {
    if (ground_truth[i] == 0.0) {
      continue;
    }

    GALOIS_LOG_VERBOSE("Truth {} input {}", ground_truth[i], input[i]);

    if (input[i] == 0.0) {
      loss -= ground_truth[i] * std::log(static_cast<GNNFloat>(1e-10));
    } else {
      loss -= ground_truth[i] * std::log(input[i]);
    }
  }

  return loss;
}

void galois::GNNCrossEntropyDerivative(const size_t vector_length,
                                       const GNNFloat* ground_truth,
                                       const GNNFloat* input,
                                       GNNFloat* gradients) {
  for (size_t i = 0; i < vector_length; i++) {
    gradients[i] = -(ground_truth[i]) / (input[i] + 1e-10);
  }
}

void galois::CBlasSGEMM(const CBLAS_TRANSPOSE trans_a,
                        const CBLAS_TRANSPOSE trans_b, size_t input_rows,
                        size_t input_columns, size_t output_columns,
                        const GNNFloat* a, const GNNFloat* b,
                        GNNFloat* output) {
  // set lead dimension based on cblas spec w.r.t. transpose setting
  size_t lead_dim_a = (trans_a == CblasNoTrans) ? input_columns : input_rows;
  size_t lead_dim_b =
      (trans_b == CblasNoTrans) ? output_columns : input_columns;
  // do the MM
  cblas_sgemm(CblasRowMajor, trans_a, trans_b, input_rows, output_columns,
              input_columns, 1.0, a, lead_dim_a, b, lead_dim_b, 0.0, output,
              output_columns);
}
