#include "math_functions.hh"
#include "context.h"

void gpu_rng_uniform(const int n, unsigned* r) {
  CURAND_CHECK(curandGenerate(Context::curand_generator(), r, n));
}

void gpu_rng_uniform(const int n, const float_t a, const float_t b,
                     float_t* r) {
  CURAND_CHECK(curandGenerateUniform(Context::curand_generator(), r, n));
  const float range = b - a;
  if (range != float_t(1))
    scal_gpu(n, range, r);
  if (a != float_t(0))
    add_scalar_gpu(n, a, r);
}

void gpu_rng_gaussian(const int n, const float_t mu, const float_t sigma,
                      float_t* r) {
  CURAND_CHECK(
      curandGenerateNormal(Context::curand_generator(), r, n, mu, sigma));
}

void out_malloc_device(int n, mask_t* h_masks, mask_t* d_masks, float_t* loss) {
  CUDA_CHECK(cudaMalloc((void**)&d_masks, n * sizeof(mask_t)));
  CUDA_CHECK(
      cudaMemcpy(d_masks, h_masks, n * sizeof(mask_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc((void**)&loss, n * sizeof(float_t)));
}

void gconv_malloc_device(size_t x, size_t y, size_t z, bool dropout,
                         unsigned* masks, float_t* in, float_t* out,
                         float_t* matrix, float_t* grad) {
  if (dropout)
    CUDA_CHECK(cudaMalloc((void**)&masks, x * y * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc((void**)&in, x * y * sizeof(float_t)));
  CUDA_CHECK(cudaMalloc((void**)&out, x * z * sizeof(float_t)));
  CUDA_CHECK(cudaMalloc((void**)&matrix, y * z * sizeof(float_t)));
  auto init_range = sqrt(6.0 / (y + z));
  // Glorot & Bengio (AISTATS 2010)
  gpu_rng_uniform(y * z, -init_range, init_range, matrix);
  CUDA_CHECK(cudaMalloc((void**)&grad, y * z * sizeof(float_t)));
  CUDA_CHECK(cudaMemset(grad, 0, y * z * sizeof(float_t)));
}

void copy_gpu(size_t len, const float_t* in, float_t* out) {
  CUDA_CHECK(
      cudaMemcpy(out, in, len * sizeof(float_t), cudaMemcpyDeviceToDevice));
}

__global__ void dropout_kernel(const int n, const float scale,
                               const float dropout_rate, const float_t* in,
                               unsigned* masks, float_t* out) {
  CUDA_KERNEL_LOOP(i, n) {
    // masks[i] = bernoulli(dropout_rate);
    out[i] = in[i] * masks[i] * scale;
  }
}

void dropout_gpu(const int n, const float scale, const float dropout_rate,
                 const float_t* in, unsigned* masks, float_t* out) {
  dropout_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, scale, dropout_rate, in, masks, out);
}

// flattern data into 1D before feed into the ReLU operater
__global__ void relu_kernel(const int n, const float_t* in, float_t* out) {
  CUDA_KERNEL_LOOP(index, n) { out[index] = in[index] > 0 ? in[index] : 0; }
}

void relu_gpu(const int n, const float_t* in, float_t* out) {
  relu_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, in, out);
}

__global__ void d_relu_kernel(const int n, const float_t* in_diff,
                              const float_t* data, float_t* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = data[index] > 0 ? in_diff[index] : 0;
  }
}

void d_relu_gpu(const int n, const float_t* in_diff, const float_t* data,
                float_t* out_diff) {
  d_relu_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, in_diff, data,
                                                          out_diff);
}

void sgemm_gpu(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
               const int M, const int N, const int K, const float alpha,
               const float* A, const float* B, const float beta, float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Context::cublas_handle(), cuTransB, cuTransA, N, M,
                           K, &alpha, B, ldb, A, lda, &beta, C, N));
}

void matmul1D1D_gpu(const size_t dim_x, const size_t dim_y, const size_t dim_z,
                    const float_t* A, const float_t* B, float_t* C) {
  const CBLAS_TRANSPOSE TransA = CblasNoTrans;
  const CBLAS_TRANSPOSE TransB = CblasNoTrans;
  sgemm_gpu(TransA, TransB, dim_x, dim_y, dim_z, 1.0, A, B, 0.0, C);
}

// the arguments of the maxima
int argmax_gpu(const size_t n, const float_t* x) { return 0; }

void gemv_gpu(const CBLAS_TRANSPOSE TransA, const int M, const int N,
              const float alpha, const float* A, const float* x,
              const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Context::cublas_handle(), cuTransA, N, M, &alpha, A,
                           N, x, 1, &beta, y, 1));
}

void scal_gpu(const int N, const float alpha, float* X) {
  CUBLAS_CHECK(cublasSscal(Context::cublas_handle(), N, &alpha, X, 1));
}

void dot_gpu(const int n, const float* x, const float* y, float* out) {
  CUBLAS_CHECK(cublasSdot(Context::cublas_handle(), n, x, 1, y, 1, out));
}

void asum_gpu(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Context::cublas_handle(), n, x, 1, y));
}

void scale_gpu(const int n, const float alpha, const float* x, float* y) {
  CUBLAS_CHECK(cublasScopy(Context::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Context::cublas_handle(), n, &alpha, y, 1));
}

__global__ void set_kernel(const int n, const float_t alpha, float_t* y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = alpha; }
}

void set_gpu(const int N, const float_t alpha, float_t* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(float_t) * N));
    return;
  }
  set_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, alpha, Y);
}

__global__ void add_scalar_kernel(const int n, const float_t alpha,
                                  float_t* y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] += alpha; }
}

void add_scalar_gpu(const int N, const float_t alpha, float_t* Y) {
  add_scalar_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, alpha, Y);
}

__global__ void vadd_kernel(const int n, const float_t* a, const float_t* b,
                            float_t* y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = a[index] + b[index]; }
}

void vadd_gpu(const int N, const float_t* a, const float_t* b, float_t* y) {
  vadd_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, a, b, y);
}

// TODO: use warp
__device__ void softmax(int n, const float_t* input, float_t* output) {
  float_t max = input[0];
  for (size_t i = 1; i < n; i++)
    if (input[i] > max)
      max = input[i];
  float_t denominator = 0.0;
  for (size_t i = 0; i < n; i++) {
    output[i] = exp(input[i] - max);
    denominator += output[i];
  }
  for (size_t i = 0; i < n; i++)
    output[i] /= denominator;
}

// TODO: use warp
__device__ void d_softmax(size_t n, const float_t* p, const float_t* dp,
                          float_t* dy) {
  for (size_t i = 0; i < n; i++) {
    dy[i] = 0;
    for (size_t j = 0; j < n; j++) {
      float_t df = (j == i) ? p[i] * (1.0 - p[i]) : -p[j] * p[i];
      dy[i] += df * dp[j];
    }
  }
}

__device__ void cross_entropy(int n, const label_t idx, const float_t* p,
                              float_t& loss) {
  if (p[idx] == 0.0)
    loss -= log(float_t(1e-10));
  else
    loss -= log(p[idx]);
}

__device__ void d_cross_entropy(int n, const label_t idx, const float_t* p,
                                float_t* d) {
  for (int i = 0; i < n; i++)
    if (i == (int)idx)
      d[i] = -1.0 / (p[i] + 1e-10);
    else
      d[i] = 0.0;
}

// n: number of vectors
// len: length of vectors
// for each vector, do softmax to normalize the vector, and then compute a loss
__global__ void softmax_cross_entropy_kernel(int n, int len,
                                             const float_t* in_data,
                                             const mask_t* masks,
                                             const label_t* labels,
                                             float_t* loss, float_t* out_data) {
  CUDA_KERNEL_LOOP(i, n) {
    if (masks[i] == 1) { // masked
      softmax(len, in_data + len * i,
              out_data + len * i); // normalize using softmax
      loss[i] = 0.0;
      cross_entropy(len, labels[i], &out_data[len * i], loss[i]);
    }
  }
}

void softmax_cross_entropy_gpu(int n, int len, const float_t* in,
                               const mask_t* masks, const label_t* labels,
                               float_t* loss, float_t* out) {
  softmax_cross_entropy_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, len, in, masks, labels, loss, out);
}

__global__ void
d_softmax_cross_entropy_kernel(int n, int len, const float_t* in,
                               const mask_t* masks, const label_t* labels,
                               const float_t* out, float_t* diff) {
  CUDA_KERNEL_LOOP(i, n) {
    float_t out_grad[41];
    d_cross_entropy(len, labels[i], out + len * i, out_grad);
    d_softmax(len, out + len * i, out_grad, diff + len * i);
  }
}

void d_softmax_cross_entropy_gpu(int n, int len, const float_t* in,
                                 const mask_t* masks, const label_t* labels,
                                 const float_t* out, float_t* diff) {
  d_softmax_cross_entropy_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, len, in, masks, labels, out, diff);
}
