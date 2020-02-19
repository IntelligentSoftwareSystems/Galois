#pragma once
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cutils.h"

// flattern data into 1D before feed into the ReLU operater
__global__ void relu_gpu(const int n, const float_t* in, float_t* out) {
	CUDA_KERNEL_LOOP(index, n) {
		out[index] = in[index] > 0 ? in[index] : 0;
	}
}

__global__ void d_relu_gpu(const int n, const float_t* in_diff, const float_t* in_data, float_t* out_diff) {
	CUDA_KERNEL_LOOP(index, n) {
		out_diff[index] = in_data[index] > 0 ? in_diff[index] : 0;
	}
}

void sgemm_gpu(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
	const int M, const int N, const int K, const float alpha, 
	const float* A, const float* B, const float beta, float* C) {
	// Note that cublas follows fortran order.
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;
	cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	CUBLAS_CHECK(cublasSgemm(cublas_handle(), cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

void gemv_gpu(const CBLAS_TRANSPOSE TransA, const int M, const int N, 
	const float alpha, const float* A, const float* x, const float beta, float* y) {
	cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
	CUBLAS_CHECK(cublasSgemv(cublas_handle(), cuTransA, N, M, &alpha, A, N, x, 1, &beta, y, 1));
}

void scal_gpu<float>(const int N, const float alpha, float *X) {
	CUBLAS_CHECK(cublasSscal(cublas_handle(), N, &alpha, X, 1));
}

