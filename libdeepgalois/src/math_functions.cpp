#include <random>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <immintrin.h>
#include "galois/Timer.h"
#include "galois/Galois.h"
#include "deepgalois/utils.h"
#include "deepgalois/random.h"
#include "deepgalois/math_functions.hh"

#ifdef USE_MKL
#include <mkl.h>
#else // If use MKL, simply include the MKL header
extern "C" {
#include <cblas.h>
}
#endif

#define NOT_IMPLEMENTED                                                        \
  do {                                                                         \
    std::cout << "Not Implemented Yet";                                        \
    exit(1);                                                                   \
  } while (0);

/*
#include <boost/random.hpp>
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

// anon namespace so these things don't leak elsewhere
namespace {
static deepgalois::PerThreadRNG* per_thread_rng = nullptr;
}

namespace deepgalois {

namespace math {

inline uint8_t bernoulli(float_t p) {
  if (!per_thread_rng) {
    per_thread_rng = new PerThreadRNG();
  }
  return per_thread_rng->get_number() > p ? 1 : 0;
}

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

#ifdef USE_MKL
void csrmm_cpu(const int M, const int N, const int K, const int,
               const float alpha, float* A_nonzeros, int* A_idx_ptr,
               int* A_nnz_idx, const float* B, const float beta, float* C) {
#else
void csrmm_cpu(const int, const int, const int, const int, const float, float*,
               int*, int*, const float*, const float, float*) {
#endif
#ifdef USE_MKL
  // mkl_set_num_threads(56);
  // const char *matdescra = "GXXCX";//6 bytes
  // const char transa = 'N';
  // mkl_scsrmm(&transa, &M , &N, &K, &alpha, matdescra, A_nonzeros, A_nnz_idx,
  // A_idx_ptr, A_idx_ptr+1, B, &N, &beta, C, &N);
  sparse_status_t status;
  bool need_trans              = false;
  bool is_row_major            = true;
  sparse_matrix_t csrA         = NULL;
  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  sparse_layout_t layout =
      (is_row_major ? SPARSE_LAYOUT_ROW_MAJOR : SPARSE_LAYOUT_COLUMN_MAJOR);
  status = mkl_sparse_s_create_csr(&csrA, indexing, M, K, A_idx_ptr,
                                   A_idx_ptr + 1, A_nnz_idx, A_nonzeros);
  if (status != SPARSE_STATUS_SUCCESS) {
    std::cout << "mkl_sparse_s_create_csr status :" << status << std::endl;
    exit(1);
  }
  sparse_operation_t transa = (need_trans ? SPARSE_OPERATION_TRANSPOSE
                                          : SPARSE_OPERATION_NON_TRANSPOSE);
  struct matrix_descr descrA;
  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
  // descrA.mode = SPARSE_FILL_MODE_UPPER;
  // descrA.diag = SPARSE_DIAG_NON_UNIT;
  // mkl_sparse_set_mm_hint(csrA, transa, descrA, layout, N, 1);
  // mkl_sparse_optimize(csrA);
  status =
      mkl_sparse_s_mm(transa, alpha, csrA, descrA, layout, B, N, N, beta, C, N);
  if (status != SPARSE_STATUS_SUCCESS) {
    std::cout << "mkl_sparse_s_create_csr status :" << status << std::endl;
    exit(1);
  }
  mkl_sparse_destroy(csrA);
#else
  NOT_IMPLEMENTED;
#endif
}

// matrix-vector multiply
void mvmul(const CBLAS_TRANSPOSE TransA, const int M, const int N,
           const float alpha, const float* A, const float* x, const float beta,
           float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

const size_t vec_len = 8; // for 32-bit floating point in AVX2; TODO AVX512

void vadd_cpu(size_t n, const float_t* a, const float_t* b, float_t* y) {
#ifdef USE_MKL
  vsAdd(n, a, b, y);
#else
#ifdef __AVX2__
  const size_t alignedN = n - n % vec_len;
  for (size_t i = 0; i < alignedN; i += vec_len)
    _mm256_storeu_ps(
        &y[i], _mm256_add_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i])));
  for (size_t i = alignedN; i < n; ++i)
    y[i] = a[i] + b[i];
#else
  for (size_t i = 0; i < n; ++i)
    y[i] = a[i] + b[i];
#endif
#endif
}

void scal(size_t n, const float_t alpha, float_t* x) {
  cblas_sscal(n, alpha, x, 1);
}

void scale(size_t n, const float_t alpha, const float_t* x, float_t* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

void axpy(size_t n, const float_t a, float_t* x, float_t* y) {
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

// l2 normalization
float_t l2_norm(size_t n, const float_t* x) { return cblas_snrm2(n, x, 1); }

// dot product
float_t dot(size_t n, const float_t* x, const float_t* y) {
  return cblas_sdot(n, x, 1, y, 1);
}

// concatenation of two vectors into one
void concat(size_t n, const float_t* x, const float_t* y, float_t* z) {
  copy_cpu(n, x, z);
  copy_cpu(n, y, z+n);
}

void clear_cpu(size_t n, float_t* in) {
  // for (size_t i = 0; i < n; i++) in[i] = 0;
  std::fill(in, in + n, 0);
  // memset(in, 0, n*sizeof(float_t));
}

void dropout(size_t m, float scale, float dropout_rate, const float_t* in,
             mask_t* masks, float_t* out) {
  for (size_t i = 0; i < m; ++i)
    masks[i] = bernoulli(dropout_rate);
  for (size_t i = 0; i < m; ++i)
    out[i] = in[i] * (float_t)masks[i] * scale;
}

void dropout_cpu(size_t n, size_t m, float scale, float dropout_rate,
                 const float_t* in, mask_t* masks, float_t* out) {
  size_t len = n * m;

  galois::do_all(
      galois::iterate((size_t)0, len),
      [&](size_t i) { masks[i] = bernoulli(dropout_rate); },
      galois::loopname("dropout RNG"));

  galois::do_all(
      galois::iterate((size_t)0, len),
      [&](const auto& i) { out[i] = in[i] * (float_t)masks[i] * scale; },
      galois::loopname("dropout"));
}

void d_dropout(size_t m, float scale, const float_t* in, mask_t* masks,
               float_t* out) {
  for (size_t i = 0; i < m; ++i)
    out[i] = in[i] * (float_t)masks[i] * scale;
}

void d_dropout_cpu(size_t n, size_t m, float scale, const float_t* in,
                   mask_t* masks, float_t* out) {
  galois::do_all(
      galois::iterate((size_t)0, n * m),
      [&](const auto& i) { out[i] = in[i] * (float_t)masks[i] * scale; },
      galois::loopname("d_dropout"));
}

void relu_cpu(size_t n, const float_t* in, float_t* out) {
  // TODO: vectorize
  galois::do_all(
      galois::iterate((size_t)0, n),
      [&](const auto& i) { out[i] = std::max(in[i], float_t(0)); },
      galois::chunk_size<64>(), galois::loopname("relu"));
}

void d_relu_cpu(size_t n, const float_t* in, const float_t* data,
                float_t* out) {
  // TODO: vectorize
  // check if original data greater than 0; if so keep grad
  galois::do_all(
      galois::iterate((size_t)0, n),
      [&](const auto& i) {
        out[i] = data[i] > float_t(0) ? in[i] : float_t(0);
      },
      galois::chunk_size<64>(), galois::loopname("d_relu"));
}

void leaky_relu(float_t epsilon, float_t in, float_t &out) {
  out = in > 0.0 ? in : epsilon * in;
}

void d_leaky_relu(float_t epsilon, float_t in, float_t data, float_t &out) {
  out = in * (data > 0.0 ? 1.0 : epsilon);
}

void leaky_relu_cpu(size_t n, float_t epsilon, const float_t* in,
                    float_t* out) {
  // TODO: vectorize
  galois::do_all(
      galois::iterate((size_t)0, n),
      [&](const auto& i) { out[i] = in[i] > 0 ? in[i] : epsilon * in[i]; },
      galois::chunk_size<64>(), galois::loopname("leaky_relu"));
}

void d_leaky_relu_cpu(size_t n, float_t epsilon, const float_t* in,
                      const float_t* data, float_t* out) {
  // TODO: vectorize
  galois::do_all(
      galois::iterate((size_t)0, n),
      [&](const auto& i) {
        out[i] = in[i] * (data[i] > float_t(0) ? float_t(1) : epsilon);
      },
      galois::chunk_size<64>(), galois::loopname("d_leaky_relu"));
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

void d_softmax(size_t n, const float_t*, const float_t* p, float_t* dy,
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

void d_cross_entropy(size_t n, const float_t* y, const float_t* p, float_t* d) {
  for (size_t i = 0; i < n; i++) {
    d[i] = -y[i] / (p[i] + float_t(1e-10));
  }
}

// use sigmoid instead of softmax for multi-class datasets, e.g. ppi, yelp and
// amazon inline float_t sigmoid_func(float_t x) { return 0.5 * tanh(0.5 * x) +
// 0.5; }
inline float_t sigmoid_func(float_t x) { return 1. / (1. + expf(-x)); }

// Sigmoid
void sigmoid(size_t n, const float_t* in, float_t* out) {
  for (size_t i = 0; i < n; i++) {
    out[i] = 1. / (1. + expf(-in[i]));
  }
}

void d_sigmoid(size_t n, const float_t*, const float_t* p, float_t* dy,
               const float_t* dp) {
  for (size_t i = 0; i < n; i++) {
    dy[i] = dp[i] * p[i] * (float_t(1) - p[i]);
  }
}

void copy_cpu(size_t n, const float_t* in, float_t* out) {
  // std::copy(in, in + n, out);
  // memcpy(out, in, sizeof(float_t) * n);
  cblas_scopy(n, in, 1, out, 1);
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

float reduce_mean(size_t n, const float_t* x) {
  float_t sum = 0.;
  for (size_t i = 0; i < n; i++) {
    sum += (float_t)x[i];
  }
  return sum / (float_t)n;
}

} // end namespace math
} // end namespace deepgalois
