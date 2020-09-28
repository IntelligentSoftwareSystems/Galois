#include <immintrin.h>
#include "galois/GNNMath.h"

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
