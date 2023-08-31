#include <algorithm>
#include <immintrin.h>
#include "galois/GNNMath.h"
#include "galois/Logging.h"

void galois::VectorZero(size_t length, GNNFloat* a) {
  for (size_t i = 0; i < length; i++) {
    a[i] = 0;
  }
}

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
  galois::gWarn("No vectorization support on this machine! Falling back to "
                "simple for loop");
  // no vector -> trivial loop add
  for (size_t i = 0; i < length; ++i) {
    output[i] = a[i] + b[i];
  }
#endif
}

void galois::VectorMulAdd(size_t length, const GNNFloat* a, const GNNFloat* b,
                          const GNNFloat b_scale, GNNFloat* output) {
#ifdef __AVX512F__
  // 512
  constexpr size_t vectorization_length = 16;
  const size_t aligned_end = length - length % vectorization_length;
  __m512 scale_vec_main    = _mm512_set_ps(
         b_scale, b_scale, b_scale, b_scale, b_scale, b_scale, b_scale, b_scale,
         b_scale, b_scale, b_scale, b_scale, b_scale, b_scale, b_scale, b_scale);
  for (size_t i = 0; i < aligned_end; i += vectorization_length) {
    _mm512_storeu_ps(
        &output[i],
        _mm512_add_ps(_mm512_loadu_ps(&a[i]),
                      _mm512_mul_ps(scale_vec_main, _mm512_loadu_ps(&b[i]))));
  }
  // handle the rest
  for (size_t i = aligned_end; i < length; ++i) {
    output[i] = a[i] + b[i] * b_scale;
  }
#else
#ifdef __AVX2__
  constexpr size_t vectorization_length =
      8; // for 32-bit floating point in AVX2; TODO AVX512
  // can only do up to a particular multiple due to alignment
  // create scale vector for b
  __m128 scale_vec_half = _mm_set_ps(b_scale, b_scale, b_scale, b_scale);
  __m256 scale_vec_main = _mm256_castps128_ps256(scale_vec_half);
  scale_vec_main = _mm256_insertf128_ps(scale_vec_main, scale_vec_half, 1);

  const size_t aligned_end = length - length % vectorization_length;
  // do add via vector ops
  for (size_t i = 0; i < aligned_end; i += vectorization_length) {
    _mm256_storeu_ps(
        &output[i],
        _mm256_add_ps(_mm256_loadu_ps(&a[i]),
                      _mm256_mul_ps(scale_vec_main, _mm256_loadu_ps(&b[i]))));
  }

  // handle the rest
  for (size_t i = aligned_end; i < length; ++i) {
    output[i] = a[i] + b[i] * b_scale;
  }
#else
  galois::gWarn("No vectorization support on this machine! Falling back to "
                "simple for loop");
  // no vector -> trivial loop add
  for (size_t i = 0; i < length; ++i) {
    output[i] = a[i] + b[i] * b_scale;
  }
#endif
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
  // TODO(loc) remove this function, unnecessary as cross/softmax derivatives
  // can be merged as currently done in Softmax code
  // will do so in a separate commit
  GALOIS_LOG_FATAL("Should not need this function anymore with simplified "
                   "combined derivatives in each layer");
  for (size_t i = 0; i < vector_length; i++) {
    for (size_t j = 0; j < vector_length; j++) {
      temp_vector[j] = (j == i) ? prev_output[i] * (1.0 - prev_output[i])
                                : -prev_output[j] * prev_output[i];
    }
    GNNFloat sdot_result = 0;
    // TODO use vector instructions? would need another loop to add everything
    // together + a temp vector to store results so probably about the same?
    for (size_t k = 0; k < vector_length; k++) {
      sdot_result += prev_output_derivative[k] * temp_vector[k];
    }
    output[i] = sdot_result;

    // TODO this is currently disabled because of a nested parallelism problem
    // (cblas may use more threads)
    // output[i] =
    //    cblas_sdot(vector_length, prev_output_derivative, 1, temp_vector, 1);
  }
}

void galois::CBlasSGEMM(const CBLAS_TRANSPOSE trans_a,
                        const CBLAS_TRANSPOSE trans_b, size_t input_rows,
                        size_t input_columns, size_t output_columns,
                        const GNNFloat* a, const GNNFloat* b,
                        GNNFloat* output) {
  CBlasSGEMM(trans_a, trans_b, input_rows, input_columns, output_columns, a, b,
             output, false);
}

void galois::CBlasSGEMM(const CBLAS_TRANSPOSE trans_a,
                        const CBLAS_TRANSPOSE trans_b, size_t input_rows,
                        size_t input_columns, size_t output_columns,
                        const GNNFloat* a, const GNNFloat* b, GNNFloat* output,
                        bool accumulate) {
  // set lead dimension based on cblas spec w.r.t. transpose setting
  size_t lead_dim_a = (trans_a == CblasNoTrans) ? input_columns : input_rows;
  size_t lead_dim_b =
      (trans_b == CblasNoTrans) ? output_columns : input_columns;
  // do the MM
  // TODO roll our own sgemm rather than use 3rd party?
  cblas_sgemm(CblasRowMajor, trans_a, trans_b, input_rows, output_columns,
              input_columns, 1.0, a, lead_dim_a, b, lead_dim_b,
              accumulate ? 1.0 : 0.0, output, output_columns);
}
