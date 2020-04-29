#include "deepgalois/math_functions.hh"
#include "galois/Timer.h"
#include "galois/Galois.h"
#include <random>
#include <immintrin.h>
#include <boost/random.hpp>
#include "deepgalois/utils.h"

#ifdef USE_MKL
#include <mkl.h>
#else  // If use MKL, simply include the MKL header
extern "C" {
#include <cblas.h>
}
#endif

#define NOT_IMPLEMENTED                \
  do {                                 \
    std::cout << "Not Implemented Yet";\
    exit(1);                           \
  } while(0);

std::default_random_engine generator;
std::uniform_real_distribution<float_t> distribution(0.0,1.0);
/*
typedef boost::mt19937 rng_t;
inline rng_t* deepgalois_rng() {
  return static_cast<rng_t*>(Context::rng_stream().generator());
}

void rng_bernoulli(size_t n, const float_t p, uint8_t* r) {
  boost::bernoulli_distribution<float_t> random_distribution(p);
  boost::variate_generator<rng_t*, boost::bernoulli_distribution<float_t> >
      variate_generator(deepgalois_rng(), random_distribution);
  for (size_t i = 0; i < n; ++i)
    r[i] = variate_generator();
}
*/
namespace deepgalois {

inline uint8_t bernoulli(float_t p) {
  //return uniform_rand(float_t(0), float_t(1)) > p ? 1 : 0;
  return distribution(generator) > p ? 1 : 0;
}

namespace math {

//! wrapper function to call cblas_sgemm
void sgemm_cpu(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
               const int M, const int N, const int K, const float alpha,
               const float* A, const float* B, const float beta, float* C) {
  galois::StatTimer Tmatmul("MatMul");
  Tmatmul.start();
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, N);
  Tmatmul.stop();
}

void csrmm_cpu(const int M, const int N, const int K, const int nnz, 
               const float alpha, const float* A_nonzeros, 
	           const int* A_idx_ptr, const int* A_nnz_idx,
               const float* B, const float beta, float* C) {
#ifdef USE_MKL
  const char *matdescra = "GXXCX";//6 bytes
  const char transa = 'N';
  //printf("Calling Intel MKL\n"); exit(1);
  mkl_scsrmm(&transa, &M , &N, &K, &alpha , matdescra,
             A_nonzeros, A_nnz_idx, A_idx_ptr, A_idx_ptr+1,
             B, &N, &beta , C, &N);
#else
  NOT_IMPLEMENTED;
#endif
}

const size_t vec_len = 8; // for 32-bit floating point in AVX2
// vector add
#if defined(__AVX__) || defined(__AVX2__)
void vadd_cpu(size_t n, const float_t* a, const float_t* b, float_t* out) {
  const size_t alignedN = n - n % vec_len;
  for (size_t i = 0; i < alignedN; i += vec_len)
    _mm256_storeu_ps(&out[i], _mm256_add_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i])));
  for (size_t i = alignedN; i < n; ++i) out[i] = a[i] + b[i];
}

void vadd(const vec_t& a, const vec_t& b, vec_t& out) {
  size_t n = out.size();
  vadd_cpu(n, &a[0], &b[0], &out[0]);
}
#else
void vadd(const vec_t& a, const vec_t& b, vec_t& out) {
  for (size_t i = 0; i < out.size(); ++i) out[i] = a[i] + b[i];
}
void vadd_cpu(size_t n, const float_t* a, const float_t* b, float_t* out) {
  for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
#endif

#if defined(__AVX__) || defined(__AVX2__)
void mul_scalar(size_t n, const float_t alpha, const float_t* in, float_t* out) {
  const size_t alignedN = n - n % vec_len;
  const __m256 scal = _mm256_set1_ps(alpha);
  for (size_t i = 0; i < alignedN; i += vec_len)
    _mm256_storeu_ps(&out[i], _mm256_mul_ps(_mm256_loadu_ps(&in[i]), scal));
  for (size_t i = alignedN; i < n; ++i) out[i] = alpha * in[i];
}

// SAXPY stands for “Single-precision A*X Plus Y"
/*
void axpy(size_t n, const float_t a, float_t *x, float_t *y) {
  const size_t alignedN = n - n % vec_len;
  const __m256 alpha = _mm256_set1_ps(a);
  for (size_t i = 0; i < alignedN; i += vec_len) {
    __m256  product = _mm256_mul_ps(_mm256_loadu_ps(&x[i]), alpha);
    _mm256_storeu_ps(&y[i], _mm256_add_ps(_mm256_loadu_ps(&y[i]), product));
  }
  for (size_t i = alignedN; i < n; ++i) y[i] = a * x[i] + y[i];
}

float_t l2_norm(size_t n, const float_t* in) {
  const size_t alignedN = n - n % vec_len;
  __m256 vsum = _mm256_set1_ps(0.0);
  for (size_t i = 0; i < alignedN; i += vec_len) {
    __m256 a = _mm256_loadu_ps(&in[i]);
    vsum = _mm256_add_ps(vsum, _mm256_mul_ps(a, a));
  }
  __m256 sum = _mm256_hadd_ps(vsum, vsum);
  return (((float_t*)&sum)[0] + ((float_t*)&sum)[2]) / 2.0;
}
*/
#else
// vector multiply scalar
void mul_scalar(const float_t alpha, vec_t& Y) {
  for (size_t i = 0; i < Y.size(); ++i) Y[i] *= alpha;
}

void mul_scalar(size_t n, const float_t alpha, const float_t* in, float_t* out) {
  for (size_t i = 0; i < n; ++i) out[i] = alpha * in[i];
}

//void axpy(size_t n, const float_t a, float_t *x, float_t *y) {
//  for (size_t i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
//}

//float_t l2_norm(size_t n, const float_t* a) {
//  float_t sum = 0.0;
//  for (size_t i = 0; i < n; ++i) sum += a[i] * a[i];
//  return sum / 2.0;
//}
#endif

void axpy(size_t n, const float_t a, float_t *x, float_t *y) {
  cblas_saxpy(n, a, x, 1, y, 1);
}

int argmax(const size_t n, const float_t* x) {
  float_t max = x[0];
  int max_ind = 0;
  for (size_t i = 1; i < n; i++) {
    if (x[i] > max) {
      max_ind = i;
      max     = x[i];
    }
  }
  return max_ind;
}

float_t l2_norm(size_t n, const float_t* x) {
  return cblas_snrm2(n, x, 1);
}

// dot product
float_t dot(const vec_t& x, const vec_t& y) {
  float_t sum = 0;
  for (size_t i = 0; i < x.size(); ++i)
    sum += x[i] * y[i];
  return sum;
}

float_t dot(size_t n, const float_t* x, const float_t* y) {
  float_t sum = 0;
  for (size_t i = 0; i < n; ++i)
    sum += x[i] * y[i];
  return sum;
}

void clear(vec_t& in) {
  for (size_t i = 0; i < in.size(); i++)
    in[i] = 0;
}

void clear_cpu(size_t n, float_t* in) {
  //for (size_t i = 0; i < n; i++) in[i] = 0;
  std::fill(in, in+n, 0);
  // memset(in, 0, n*sizeof(float_t));
}

void dropout(size_t m, float scale, float dropout_rate, 
             const float_t* in, mask_t* masks, float_t* out) {
  for (size_t i = 0; i < m; ++i)
    masks[i] = deepgalois::bernoulli(dropout_rate);
  for (size_t i = 0; i < m; ++i)
    out[i] = in[i] * (float_t)masks[i] * scale;
}

void dropout_cpu(size_t n, size_t m, float scale, float dropout_rate,
             const float_t* in, mask_t* masks, float_t* out) {
  for (size_t i = 0; i < n*m; ++i)
    masks[i] = deepgalois::bernoulli(dropout_rate);
  galois::do_all(galois::iterate((size_t)0, n*m), [&](const auto& i) {
    out[i] = in[i] * (float_t)masks[i] * scale;
  }, galois::loopname("dropout"));
}

void d_dropout(size_t m, float scale, const float_t* in, mask_t* masks, float_t* out) {
  for (size_t i = 0; i < m; ++i)
    out[i] = in[i] * (float_t)masks[i] * scale;
}

void d_dropout_cpu(size_t n, size_t m, float scale, const float_t* in,
                   mask_t* masks, float_t* out) {
  galois::do_all(galois::iterate((size_t)0, n*m), [&](const auto& i) {
    out[i] = in[i] * (float_t)masks[i] * scale;
  }, galois::loopname("d_dropout"));
}

void relu_cpu(size_t n, const float_t* in, float_t* out) {
  // TODO: vectorize
  galois::do_all(galois::iterate((size_t)0, n), [&](const auto& i) {
    out[i] = std::max(in[i], float_t(0));
  }, galois::chunk_size<64>(), galois::loopname("relu"));
}

void d_relu_cpu(size_t n, const float_t* in, const float_t* data, float_t* out) {
  // TODO: vectorize
  // check if original data greater than 0; if so keep grad
  galois::do_all(galois::iterate((size_t)0, n), [&](const auto& i) {
    out[i] = data[i] > float_t(0) ? in[i] : float_t(0);
  }, galois::chunk_size<64>(), galois::loopname("d_relu"));
}

void leaky_relu_cpu(size_t n, float_t epsilon, const float_t* in, float_t* out) {
  // TODO: vectorize
  galois::do_all(galois::iterate((size_t)0, n), [&](const auto& i) {
    out[i] = in[i] > 0 ? in[i] : epsilon * in[i];
  }, galois::chunk_size<64>(), galois::loopname("leaky_relu"));
}

void d_leaky_relu_cpu(size_t n, float_t epsilon, const float_t* in, 
                      const float_t* data, float_t* out) {
  // TODO: vectorize
  galois::do_all(galois::iterate((size_t)0, n), [&](const auto& i) {
    out[i] = in[i] * (data[i] > float_t(0) ? float_t(1) : epsilon);
  }, galois::chunk_size<64>(), galois::loopname("d_leaky_relu"));
}

void softmax(const vec_t& input, vec_t& output) {
  const float_t max = *std::max_element(input.begin(), input.end());
  float_t denominator(0);
  for (size_t i = 0; i < input.size(); i++) {
    output[i] = std::exp(input[i] - max);
    denominator += output[i];
  }
  for (size_t i = 0; i < input.size(); i++)
    output[i] /= denominator;
}

void softmax(size_t n, const float_t* input, float_t* output) {
  const float_t max = *std::max_element(input, input + n);
  float_t denominator(0);
  for (size_t i = 0; i < n; i++) {
    output[i] = std::exp(input[i] - max);
    denominator += output[i];
  }
  for (size_t i = 0; i < n; i++)
    output[i] /= denominator;
}

void d_softmax(const vec_t& y, const vec_t& p, vec_t& dy, const vec_t& dp) {
  auto n = y.size();
  vec_t df(n, 0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      // float_t delta_ij = i == j? 1 : 0;
      // df[i] += p[j] * (delta_ij - p[i]);
      df[j] = (j == i) ? p[i] * (float_t(1) - p[i]) : -p[j] * p[i];
    }
    // dy = dp * (gradient of softmax)
    dy[i] = dot(dp, df);
  }
}

void d_softmax(size_t n, const float_t* y, const float_t* p, float_t* dy,
               const float_t* dp) {
  vec_t df(n, 0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      df[j] = (j == i) ? p[i] * (float_t(1) - p[i]) : -p[j] * p[i];
    }
    dy[i] = dot(n, dp, &df[0]);
  }
}

// cross-entropy loss function for multi-class classification
// y: ground truth
// p: predicted probability
float_t cross_entropy(const vec_t& y, const vec_t& p) {
  auto n = y.size();
  assert(n > 0);
  float_t loss = 0.0;
  for (size_t i = 0; i < n; i++) {
    if (y[i] == float_t(0))
      continue;
    if (p[i] == float_t(0))
      loss -= y[i] * std::log(float_t(1e-10));
    else loss -= y[i] * std::log(p[i]);
  }
  return loss;
}

float_t cross_entropy(size_t n, const float_t* y, const float_t* p) {
  float_t loss = 0.0;
  for (size_t i = 0; i < n; i++) {
    if (y[i] == float_t(0))
      continue;
    if (p[i] == float_t(0))
      loss -= y[i] * std::log(float_t(1e-10));
    else
      loss -= y[i] * std::log(p[i]);
  }
  return loss;
}

void d_cross_entropy(const vec_t& y, const vec_t& p, vec_t& d) {
  auto n = y.size();
  for (size_t i = 0; i < n; i++) {
    d[i] = -y[i] / (p[i] + float_t(1e-10));
  }
}

void d_cross_entropy(size_t n, const float_t* y, const float_t* p, float_t* d) {
  for (size_t i = 0; i < n; i++) {
    d[i] = -y[i] / (p[i] + float_t(1e-10));
  }
}

// use sigmoid instead of softmax for multi-class datasets, e.g. ppi, yelp and amazon
// inline float_t sigmoid_func(float_t x) { return 0.5 * tanh(0.5 * x) + 0.5; }
inline float_t sigmoid_func(float_t x) { return 1./(1.+expf(-x)); }

// Sigmoid
void sigmoid(const vec_t& in, vec_t &out) {
  for (size_t i = 0; i < in.size(); ++i)
    out[i] = sigmoid_func(in[i]);
}

void sigmoid(size_t n, const float_t* in, float_t* out) {
  for (size_t i = 0; i < n; i++) {
    out[i] = 1. / (1. + expf(-in[i]));
  }
}

void d_sigmoid(size_t n, const float_t* y, const float_t* p, float_t* dy, const float_t* dp) {
  for (size_t i = 0; i < n; i++) {
    dy[i] = dp[i] * p[i] * (float_t(1) - p[i]);
  }
}

void copy1D1D(const vec_t& in, vec_t& out) {
  std::copy(in.begin(), in.end(), &out[0]);
}

void copy_cpu(size_t len, const float_t* in, float_t* out) {
  std::copy(in, in + len, out);
}

// num rows in A, C; num columns in B, C; num columns in A, rows in B
void matmul1D1D(const size_t dim_x, const size_t dim_y, const size_t dim_z,
                const float_t* A, const float_t* B, float_t* C) {
  sgemm_cpu(CblasNoTrans, CblasNoTrans, dim_x, dim_y, dim_z, 1.0, A, B, 0.0, C);
}

// TODO make parallel
void transpose(size_t x, size_t y, const float_t* in, float_t* out) {
  for (size_t i = 0; i < y; i++) {
    for (size_t j = 0; j < x; j++) {
      out[i * x + j] = in[j * y + i];
    }
  }
}

// matrix-vector multiply
void mvmul(size_t m, size_t n, const float_t *matrix, const float_t *in_vector, float_t *out_vector) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      out_vector[i] += matrix[i * n + j] * in_vector[j];
    }
  }
}

float reduce_mean(size_t n, const float_t* x) {
  float_t sum = 0.;
  for (size_t i = 0; i < n; i++) {
    sum += (float_t)x[i];
  }
  return sum / (float_t)n;
}

} // end namespace math
} // end namespace deepgalois

