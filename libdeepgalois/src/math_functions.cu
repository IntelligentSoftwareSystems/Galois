#include "deepgalois/math_functions.hh"
#include "deepgalois/context.h"
#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include <curand_kernel.h>

__global__ void init_const_kernel(int n, float_t value, float_t *array) {
  CUDA_KERNEL_LOOP(i, n) { array[i] = value; }
}

void init_const_gpu(int n, float_t value, float_t *array) {
  init_const_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, value, array);
  CudaTest("solving init_const kernel failed");
}

__global__ void isnan_test(const int n, const float *data, bool *result) {
	CUDA_KERNEL_LOOP(i, n) { if (isnan(data[i])) *result = true; }
}

bool isnan_gpu(int n, const float_t *array) {
  bool  *d_result, h_result = false;
  cudaMalloc((void **)&d_result, sizeof (bool));
  cudaMemcpy(d_result, &h_result, sizeof(bool), cudaMemcpyHostToDevice);
  isnan_test<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, array, d_result);
  CudaTest("solving init_const kernel failed");
  cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
  return h_result;
}

void gpu_rng_uniform(const int n, unsigned* r) {
  CURAND_CHECK(curandGenerate(deepgalois::Context::curand_generator(), r, n));
}

void rng_uniform_gpu(const int n, const float_t a, const float_t b, float_t* r) {
  CURAND_CHECK(curandGenerateUniform(deepgalois::Context::curand_generator(), r, n));
  const float range = b - a;
  if (range != float_t(1))
    scal_gpu(n, range, r);
  if (a != float_t(0))
    add_scalar_gpu(n, a, r);
}

void gpu_rng_gaussian(const int n, const float_t mu, const float_t sigma, float_t* r) {
  CURAND_CHECK(curandGenerateNormal(deepgalois::Context::curand_generator(), r, n, mu, sigma));
}

bool is_allocated_device(float_t* data) {
  if (data == NULL) return false;
  cudaPointerAttributes attributes;
  CUDA_CHECK(cudaPointerGetAttributes(&attributes, data));
  if (attributes.devicePointer != NULL) return true;
  return false;
}

void float_malloc_device(int n, float_t*& ptr) {
  CUDA_CHECK(cudaMalloc((void**)&ptr, n * sizeof(float_t)));
}

void float_free_device(float_t*& ptr) {
  CUDA_CHECK(cudaFree(ptr));
}

void float_copy_device(int n, float_t* h_ptr, float_t *d_ptr) {
  CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, n * sizeof(float_t), cudaMemcpyHostToDevice));
}

void copy_masks_device(int n, mask_t* h_masks, mask_t*& d_masks) {
  assert(h_masks != NULL);
  CUDA_CHECK(cudaMalloc((void**)&d_masks, n * sizeof(mask_t)));
  CUDA_CHECK(cudaMemcpy(d_masks, h_masks, n * sizeof(mask_t), cudaMemcpyHostToDevice));
}

__global__ void setup_curand_kernel(const int n, curandState* state) {
  CUDA_KERNEL_LOOP(i, n) {
    // curand_init(1234, i, 0, &state[i]); // Each thread gets same seed 1234
    curand_init(7 + i, i, 0, &state[i]); // Each thread gets different seed
  }
}

__global__ void dropout_kernel(const int n, const float scale,
                               const float threshold, const float_t* in,
                               unsigned* masks, float_t* out) {
  CUDA_KERNEL_LOOP(i, n) { out[i] = in[i] * (masks[i] > threshold) * scale; }
}

void dropout_gpu(const int n, const float scale, const float dropout_rate,
                 const float_t* in, unsigned* masks, float_t* out) {
  gpu_rng_uniform(n, masks);
  //std::cout << "[debug]: dropout_gpu\n";
  dropout_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, scale, dropout_rate, in, masks, out);
  CudaTest("solving dropout kernel failed");
  //std::cout << "[debug]: dropout_gpu done\n";
}

__global__ void d_dropout_kernel(const int n, const float scale,
                                 const float threshold, const float_t* in,
                                 const unsigned* masks, float_t* out) {
  CUDA_KERNEL_LOOP(i, n) { out[i] = in[i] * (masks[i] > threshold) * scale; }
}

void d_dropout_gpu(const int n, const float scale, const float dropout_rate, 
                   const float_t* in, const unsigned* masks, float_t* out) {
  d_dropout_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, scale, dropout_rate, in, masks, out);
  CudaTest("solving d_dropout kernel failed");
}

// flattern data into 1D before feed into the ReLU operater
__global__ void relu_kernel(const int n, const float_t* in, float_t* out) {
  CUDA_KERNEL_LOOP(i, n) { out[i] = in[i] > 0 ? in[i] : 0; }
}

void relu_gpu(const int n, const float_t* in, float_t* out) {
  relu_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, in, out);
  CudaTest("solving relu kernel failed");
}

__global__ void d_relu_kernel(const int n, const float_t* in_diff,
                              const float_t* data, float_t* out_diff) {
  CUDA_KERNEL_LOOP(i, n) { out_diff[i] = data[i] > 0 ? in_diff[i] : 0; }
}

void d_relu_gpu(const int n, const float_t* in_diff, const float_t* data,
                float_t* out_diff) {
  d_relu_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, in_diff, data,
                                                          out_diff);
  CudaTest("solving d_relu kernel failed");
}

// flattern data into 1D before feed into the ReLU operater
__global__ void leaky_relu_kernel(const int n, const float_t epsilon,
                                  const float_t* in, float_t* out) {
  CUDA_KERNEL_LOOP(i, n) { out[i] = in[i] > 0 ? in[i] : epsilon * in[i]; }
}

void leaky_relu_gpu(const int n, const float_t epsilon, 
                    const float_t* in, float_t* out) {
  leaky_relu_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, epsilon, in, out);
  CudaTest("solving leaky_relu kernel failed");
}

__global__ void d_leaky_relu_kernel(const int n, const float_t epsilon, 
    const float_t* in_diff, const float_t* data, float_t* out_diff) {
  CUDA_KERNEL_LOOP(i, n) {
    out_diff[i] = in_diff[i] * (data[i] > 0 ? 1.0 : epsilon);
  }
}

void d_leaky_relu_gpu(const int n, const float_t epsilon, const float_t* in_diff, 
                      const float_t* data, float_t* out_diff) {
  d_leaky_relu_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, epsilon, in_diff, data, out_diff);
  CudaTest("solving d_leaky_relu kernel failed");
}

__global__ void matmul_kernel(int x, int y, int z, const float_t* A,
                              const float_t* B, float_t* C) {
	int row = blockIdx.x*blockDim.x+threadIdx.x;
	int col = blockIdx.y*blockDim.y+threadIdx.y;
	float_t sum = 0.0f;
	if (row < x && col < y) {
		for (int i = 0; i < z; i++) {
			sum += A[row * z + i] * B[i * y + col];
		}
	}
	C[row * y + col] = sum;
}

#define TILE_SZ 16
void matmul_gpu(const size_t x, const size_t y, const size_t z,
                    const float_t* A, const float_t* B, float_t* C) {
  dim3 threadsPerBlock(TILE_SZ, TILE_SZ);
  dim3 blocksPerGrid((y-1)/TILE_SZ+1, (x-1)/TILE_SZ+1);
  matmul_kernel<<<blocksPerGrid,threadsPerBlock>>>(x, y, z, A, B, C);
  CudaTest("solving matmul kernel failed");
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
  CUBLAS_CHECK(cublasSgemm(deepgalois::Context::cublas_handle(), cuTransB, cuTransA,
                           N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

void matmul1D1D_gpu(const size_t dim_x, const size_t dim_y, const size_t dim_z,
                    const float_t* A, const float_t* B, float_t* C) {
  const CBLAS_TRANSPOSE TransA = CblasNoTrans;
  const CBLAS_TRANSPOSE TransB = CblasNoTrans;
  sgemm_gpu(TransA, TransB, dim_x, dim_y, dim_z, 1.0, A, B, 0.0, C);
}

// C = A x B, where A is a sparse matrix in CSR format, B is the dense matrix for vertex
// feature tensor. However, since cusparse only supports column-major, while feature 
// tensor is stored in row-major, the actual computation is: C = trans(A x trans(B)).
// Currently, we use cublasSgeam to implement transposition and allocate intermediate
// workspace memory (transpose_C) for this.
void csrmm_gpu(const int M, const int N, const int K, const int nnz, 
               const float alpha, const float* A_nonzeros, 
	           const int* A_idx_ptr, const int* A_nnz_idx,
               const float* B, const float beta, float *transpose_C, float* C) {
  //std::cout << "[debug] csrmm_gpu m=" << M << ", n=" << N << ", k=" << K << ", nnz=" << nnz << "\n";
  CUSPARSE_CHECK(cusparseScsrmm2(deepgalois::Context::cusparse_handle(),
                 CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                 M, N, K, nnz, &alpha, deepgalois::Context::cusparse_matdescr(), A_nonzeros, 
                 A_idx_ptr, A_nnz_idx, B, N, &beta, transpose_C, M)); 
  //transpose C
  const float one = 1.0;
  const float zero = 0.0; 
  CUBLAS_CHECK(cublasSgeam(deepgalois::Context::cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_T,
                           N, M, &one, transpose_C, M, &zero, NULL, M, C, N)); 
}
/*
void csrmm_gpu_new(const int M, const int N, const int K, const int nnz, 
               const float alpha, const float* A_nonzeros, 
	           const int* A_idx_ptr, const int* A_nnz_idx,
               const float* B, const float beta, float *transpose_C, float* C) {
  std::cout << "[debug]: csrmm_gpu\n";
  cusparseSpMatDescr_t A_descr;
  CUSPARSE_CHECK(cusparseCreateCsr(&A_descr, M, K, nnz, A_idx_ptr, A_nnz_idx, A_nonzeros,
   	             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  cusparseDnMatDescr_t B_descr;
  CUSPARSE_CHECK(cusparseCreateDnMat(&B_descr, K, N, K, B, CUDA_R_32F, CUSPARSE_ORDER_COL));
  cusparseDnMatDescr_t C_descr;
  CUSPARSE_CHECK(cusparseCreateDnMat(&C_descr, M, N, M, C, CUDA_R_32F, CUSPARSE_ORDER_COL));
  size_t bufferSize;
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(deepgalois::Context::cusparse_handle(),
                       CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                       (void*)&alpha, A_descr, B_descr, (void*)&beta, C_descr,
                       CUDA_R_32F, CUSPARSE_COOMM_ALG1, &bufferSize));
  cudaDeviceSynchronize();
  void* buffer = NULL;
  if (bufferSize > 0) CUDA_CHECK(cudaMalloc(&buffer, bufferSize));
  CUSPARSE_CHECK(cusparseSpMM(deepgalois::Context::cusparse_handle(),
                 CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                 (const void*)&alpha, A_descr, B_descr, (const void*)&beta, C_descr, 
                 CUDA_R_32F, CUSPARSE_COOMM_ALG1, buffer));
  cudaDeviceSynchronize();
  //transpose C
  const float one = 1.0;
  const float zero = 0.0; 
  CUBLAS_CHECK(cublasSgeam(deepgalois::Context::cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_T,
                           N, M, &one, transpose_C, M, &zero, NULL, M, C, N)); 
}
//*/
void gemv_gpu(const CBLAS_TRANSPOSE TransA, const int M, const int N,
              const float alpha, const float* A, const float* x,
              const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(deepgalois::Context::cublas_handle(), cuTransA, N, M, &alpha, A,
                           N, x, 1, &beta, y, 1));
}

void scal_gpu(const int N, const float alpha, float* X) {
  CUBLAS_CHECK(cublasSscal(deepgalois::Context::cublas_handle(), N, &alpha, X, 1));
}

void dot_gpu(const int n, const float* x, const float* y, float* out) {
  CUBLAS_CHECK(cublasSdot(deepgalois::Context::cublas_handle(), n, x, 1, y, 1, out));
}

void asum_gpu(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(deepgalois::Context::cublas_handle(), n, x, 1, y));
}

void scale_gpu(const int n, const float alpha, const float* x, float* y) {
  CUBLAS_CHECK(cublasScopy(deepgalois::Context::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(deepgalois::Context::cublas_handle(), n, &alpha, y, 1));
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
  CudaTest("solving set kernel failed");
}

__global__ void add_scalar_kernel(const int n, const float_t alpha,
                                  float_t* y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] += alpha; }
}

void add_scalar_gpu(const int N, const float_t alpha, float_t* Y) {
  add_scalar_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, alpha, Y);
  CudaTest("solving add_scalar kernel failed");
}

__global__ void vadd_kernel(const int n, const float_t* a, const float_t* b,
                            float_t* y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = a[index] + b[index]; }
}

void copy_gpu(int len, const float_t* in, float_t* out) {
  CUDA_CHECK(cudaMemcpy(out, in, len * sizeof(float_t), cudaMemcpyDeviceToDevice));
}

void vadd_gpu(const int N, const float_t* a, const float_t* b, float_t* y) {
  vadd_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, a, b, y);
  CudaTest("solving vadd kernel failed");
}

// TODO: use warp
__device__ void softmax_device(int n, const float_t* input, float_t* output) {
  float_t max = input[0];
  for (int i = 1; i < n; i++)
    if (input[i] > max)
      max = input[i];
  float_t denominator = 0.0;
  for (int i = 0; i < n; i++) {
    output[i] = expf(input[i] - max);
    denominator += output[i];
	if (output[i] < 0.0) printf("in[%d]=%f, out[%d]=%f\n", i, input[i], i, output[i]);
    //assert(output[i] >= 0.0);
  }
  assert(denominator != 0.0);
  for (int i = 0; i < n; i++) {
    output[i] /= denominator;
    //assert(output[i] >= 0.0);
    //assert(output[i] <= 1.0);
  }
}

__device__ void sigmoid_device(int n, const float_t* in, float_t* out) {
  for (int i = 0; i < n; i++)
    out[i] = 1. / (1. + expf(-in[i]));
}

__device__ void cross_entropy_device(int n, const label_t idx, const float_t* p, float_t& loss) {
  if (p[idx] == 0.0) loss -= logf(float_t(1e-10));
  else loss -= logf(p[idx]);
}

// y: ground truth
// p: predictions
__device__ void cross_entropy_multi_device(int n, const label_t *y, const float_t* p, float_t& loss) {
  for (int i = 0; i < n; i++) {
    if (y[i] == 0) continue;
    if (p[i] == float_t(0)) loss -= logf(float_t(1e-10)); // avoid NaN exception
    else loss -= logf(p[i]);
  }
}

// n: number of vectors
// len: length of vectors
// for each vector, do softmax to normalize the vector, and then compute a loss
__global__ void softmax_cross_entropy_kernel(int len, int begin, int end,
                                             const float_t* in_data,
                                             const mask_t* masks,
                                             const label_t* labels,
                                             float_t* loss, float_t* out_data) {
  CUDA_KERNEL_LOOP(i, end-begin) {
    int id = begin + i;
    if (masks[id] == 1) { // masked
	  // normalize using softmax
      softmax_device(len, in_data + len*id, out_data + len*id);
      //loss[id] = 0.0;
      cross_entropy_device(len, labels[id], out_data + len*id, loss[id]);
    }
  }
}

void softmax_cross_entropy_gpu(int len, int begin, int end, const float_t* in,
                               const mask_t* masks, const label_t* labels,
                               float_t* loss, float_t* out) {
  softmax_cross_entropy_kernel<<<CUDA_GET_BLOCKS(end-begin), CUDA_NUM_THREADS>>>(
      len, begin, end, in, masks, labels, loss, out);
  CudaTest("solving softmax_cross_entropy kernel failed");
}

// n: number of vectors
// len: length of vectors
// for each vector, do softmax to normalize the vector, and then compute a loss
__global__ void sigmoid_cross_entropy_kernel(int len, int begin, int end,
                                             const float_t* in_data,
                                             const mask_t* masks,
                                             const label_t* labels,
                                             float_t* loss, float_t* out_data) {
  CUDA_KERNEL_LOOP(i, end-begin) {
    int id = begin + i;
    if (masks[id] == 1) { // masked
      sigmoid_device(len, in_data + len*id, out_data + len*id);
      cross_entropy_multi_device(len, labels, out_data + len*id, loss[id]);
    }
  }
}

void sigmoid_cross_entropy_gpu(int len, int begin, int end, const float_t* in,
                               const mask_t* masks, const label_t* labels,
                               float_t* loss, float_t* out) {
  sigmoid_cross_entropy_kernel<<<CUDA_GET_BLOCKS(end-begin), CUDA_NUM_THREADS>>>(
      len, begin, end, in, masks, labels, loss, out);
  CudaTest("solving sigmoid_cross_entropy kernel failed");
}

__device__ void d_cross_entropy_device(int n, const label_t idx, const float_t* p, float_t* d) {
  for (int i = 0; i < n; i++) {
    if (i == (int)idx) d[i] = -1.0 / (p[i] + 1e-10);
    else d[i] = 0.0;
  }
}

__global__ void d_cross_entropy_kernel(int len, int begin, int end,
                                const mask_t* masks, const label_t* labels,
                                const float_t* data, float_t* grad) {
  int base = begin * len;
  CUDA_KERNEL_LOOP(i, (end-begin)*len) {
    int id = begin + i/len;
    if (masks[id] == 1) { // masked
      if (i%len == (int)labels[id]) grad[i] = -1.0 / (data[i+base] + 1e-10);
      else grad[i] = 0.0;
      //d_cross_entropy_device(len, labels[id], data + len*id, grad + len*i);
    }
  }
} 

__global__ void d_cross_entropy_warp(int len, int begin, int end,
                                const mask_t* masks, const label_t* labels,
                                const float_t* data, float_t* grad) {
  __shared__ float_t p[BLOCK_SIZE/WARP_SIZE][MAX_NUM_CLASSES];
  const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
  const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

  for (int wid = warp_id; wid < end-begin; wid += num_warps) {
    int id = begin + wid;
    int base = id * len;	
    if (masks[id] == 1) {
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len) p[warp_lane][pid] = data[base+pid];
      }
      __syncthreads();
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len) {
          if (pid == (int)labels[id])
            grad[wid*len+pid] = -1.0 / (p[warp_lane][pid] + 1e-10);
          else grad[wid*len+pid] = 0.0;
        }
      }
    }
  }
}

__device__ void d_softmax_device(int n, const float_t* p, const float_t* dp, float_t* dy) {
  for (int i = 0; i < n; i++) {
    dy[i] = 0;
    for (int j = 0; j < n; j++) {
      float_t df = (j == i) ? p[i] * (1.0 - p[i]) : -p[j] * p[i];
      dy[i] += df * dp[j];
    }
  }
}

__global__ void d_softmax_kernel(int len, int begin, int end,
                                 const mask_t* masks, const float_t* data,
                                 const float_t* in_grad, float_t* out_grad) {
  CUDA_KERNEL_LOOP(i, end-begin) {
    int id = begin + i;
    if (masks[id] == 1) { // masked
      d_softmax_device(len, data + len*id, in_grad + len*i, out_grad + len*id);
    }
  }
} 

__global__ void d_softmax_warp(int len, int begin, int end,
                               const mask_t* masks, const float_t* data,
                               const float_t* in_grad, float_t* out_grad) {
  __shared__ float_t p[BLOCK_SIZE/WARP_SIZE][MAX_NUM_CLASSES];
  __shared__ float_t d[BLOCK_SIZE/WARP_SIZE][MAX_NUM_CLASSES];
  const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
  const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

  for (int wid = warp_id; wid < end-begin; wid += num_warps) {
    int id = begin + wid;
    int base = id * len;	
    if (masks[id] == 1) {
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len) {
          p[warp_lane][pid] = data[base+pid];
          d[warp_lane][pid] = in_grad[wid*len+pid];
        }
      }
      __syncthreads();
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len) {
          float_t sum = 0.0;
          float_t self = p[warp_lane][pid];
          for (int j = 0; j < len; j++) {
            float_t df = (j == pid) ? self * (1.0 - self) : -p[warp_lane][j] * self;
            sum += df * d[warp_lane][j];
          }
          out_grad[base+pid] = sum;
        }
      }
      __syncthreads();
    }
  }
}

__global__ void d_softmax_cross_entropy_kernel(int len, int begin, int end,
                                               const mask_t* masks, const label_t* labels,
                                               const float_t* out, float_t* diff) {
  CUDA_KERNEL_LOOP(i, end-begin) {
    int id = begin + i;
    if (masks[id] == 1) { // masked
	  float_t out_grad[41]; // TODO
      d_cross_entropy_device(len, labels[id], out + len*id, out_grad);
      d_softmax_device(len, out + len*id, out_grad, diff + len*id);
    }
  }
}

__global__ void d_softmax_cross_entropy_warp(int len, int begin, int end,
                                const mask_t* masks, const label_t* labels,
                                const float_t* data, float_t* grad) {
  __shared__ float_t p[BLOCK_SIZE/WARP_SIZE][MAX_NUM_CLASSES];
  __shared__ float_t d[BLOCK_SIZE/WARP_SIZE][MAX_NUM_CLASSES];
  const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
  const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

  for (int wid = warp_id; wid < end-begin; wid += num_warps) {
    int id = begin + wid;
    int base = id * len;	
    if (masks[id] == 1) {
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len) p[warp_lane][pid] = data[base+pid];
      }
      __syncthreads();

      // cross entropy derivative
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len) {
          if (pid == (int)labels[id])
            d[warp_lane][pid] = -1.0 / (p[warp_lane][pid] + 1e-10);
          else d[warp_lane][pid] = 0.0;
        }
      }
      __syncthreads();

      // softmax derivative
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len) {
          float_t sum = 0.0;
          float_t self = p[warp_lane][pid];
          for (int j = 0; j < len; j++) {
            float_t df = (j == pid) ? self * (1.0 - self) : -p[warp_lane][j] * self;
            sum += df * d[warp_lane][j];
          }
          grad[base+pid] = sum;
        }
      }
      __syncthreads();
    }
  }
}

void d_softmax_cross_entropy_gpu(int len, int begin, int end,
                                 const mask_t* masks, const label_t* labels,
                                 const float_t* out, float_t* diff) {
//  d_softmax_cross_entropy_kernel<<<CUDA_GET_BLOCKS(end-begin), CUDA_NUM_THREADS>>>(
//      len, begin, end, masks, labels, out, diff);
//  CudaTest("solving d_softmax_cross_entropy kernel failed");
  //float_t *grad;
  //float_malloc_device((end-begin)*len, grad);
  //d_cross_entropy_kernel<<<CUDA_GET_BLOCKS((end-begin)*len), CUDA_NUM_THREADS>>>(
  //d_cross_entropy_warp<<<(end-begin-1)/WARPS_PER_BLOCK+1, BLOCK_SIZE>>>(
  //    len, begin, end, masks, labels, out, grad);
  //CudaTest("solving d_cross_entropy kernel failed");
  //d_softmax_kernel<<<CUDA_GET_BLOCKS(end-begin), CUDA_NUM_THREADS>>>(
  //d_softmax_warp<<<(end-begin-1)/WARPS_PER_BLOCK+1, BLOCK_SIZE>>>(
  //    len, begin, end, masks, out, grad, diff);
  //CudaTest("solving d_softmax kernel failed");
  d_softmax_cross_entropy_warp<<<(end-begin-1)/WARPS_PER_BLOCK+1, BLOCK_SIZE>>>(
      len, begin, end, masks, labels, out, diff);
  CudaTest("solving d_softmax_cross_entropy_warp kernel failed");
}

__global__ void d_sigmoid_cross_entropy_warp(int len, int begin, int end,
                                             const mask_t* masks, const label_t* labels,
                                             const float_t* data, float_t* grad) {
  __shared__ float_t p[BLOCK_SIZE/WARP_SIZE][MAX_NUM_CLASSES];
  __shared__ float_t d[BLOCK_SIZE/WARP_SIZE][MAX_NUM_CLASSES];
  const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
  const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

  for (int wid = warp_id; wid < end-begin; wid += num_warps) {
    int id = begin + wid;
    int base = id * len;	
    if (masks[id] == 1) {
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len) p[warp_lane][pid] = data[base+pid];
      }
      __syncthreads();

      // cross entropy derivative
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len) {
          //if (p[warp_lane][pid] == 0)
            d[warp_lane][pid] = -(float_t)labels[base+pid] / (p[warp_lane][pid] + 1e-10);
          //else d[warp_lane][pid] = -(float_t)labels[pid] / 1e-10;
        }
      }
      __syncthreads();

      // sigmoid derivative
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len) {
          float_t self = p[warp_lane][pid];
          float_t dp = d[warp_lane][pid];
          grad[base+pid] = dp * self * (float_t(1) - self);
        }
      }
      __syncthreads();
    }
  }
}

void d_sigmoid_cross_entropy_gpu(int len, int begin, int end,
                                 const mask_t* masks, const label_t* labels,
                                 const float_t* out, float_t* diff) {
  d_sigmoid_cross_entropy_warp<<<(end-begin-1)/WARPS_PER_BLOCK+1, BLOCK_SIZE>>>(
      len, begin, end, masks, labels, out, diff);
  CudaTest("solving d_sigmoid_cross_entropy_warp kernel failed");
}

__global__ void masked_avg_loss_kernel(int begin, int end, mask_t* masks,
                                       float_t* loss, HGAccumulator<acc_t> total) {
  total.thread_entry();
  __shared__ cub::BlockReduce<acc_t, CUDA_NUM_THREADS>::TempStorage local_loss;
  CUDA_KERNEL_LOOP(i, end - begin) {
    if (masks[begin + i] == 1)
      total.reduce(loss[begin + i]);
  }
  total.thread_exit<cub::BlockReduce<acc_t, CUDA_NUM_THREADS>>(local_loss);
}

//acc_t masked_avg_loss(int begin, int end, int count, mask_t* masks, float_t* loss);
acc_t masked_avg_loss_gpu(int begin, int end, int count, mask_t* masks, float_t* loss) {
  assert(count > 0);
  HGAccumulator<acc_t> loss_accum;
  Shared<acc_t> total_loss   = Shared<acc_t>(1);
  *(total_loss.cpu_wr_ptr()) = 0;
  loss_accum.rv              = total_loss.gpu_wr_ptr();
  masked_avg_loss_kernel<<<CUDA_GET_BLOCKS(end - begin), CUDA_NUM_THREADS>>>(
      begin, end, masks, loss, loss_accum);
  CudaTest("solving masked_avg_loss kernel failed");
  cudaDeviceSynchronize();
  return *(total_loss.cpu_rd_ptr()) / count;
}

acc_t l2_norm_gpu(int n, float_t * tensor) {
  acc_t sum = 0.0;
  return sum / 2.0;
}

