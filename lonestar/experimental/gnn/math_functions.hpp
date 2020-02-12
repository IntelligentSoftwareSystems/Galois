#ifndef _MATH_FUNCTIONS_
#define _MATH_FUNCTIONS_
#include <cmath>
#include "utils.h"
#include <immintrin.h>

#ifdef WITH_BLAS
extern "C" {
#include <cblas.h>
//#include <clapack.h>
}
#endif

const float negative_slope = 0;

// vector add
template <typename DataTy = float>
inline void vadd(const std::vector<DataTy> &a, const std::vector<DataTy> &b, std::vector<DataTy> &out) {
	//for (size_t i = 0; i < out.size(); ++i) out[i] = a[i] + b[i];
	size_t n = out.size();
	size_t vec_len = 8;
	const size_t alignedN = n - n % vec_len;
	for (size_t i = 0; i < alignedN; i += vec_len)
		_mm256_storeu_ps(&out[i], _mm256_add_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i])));
	for (size_t i = alignedN; i < n; ++i) out[i] = a[i] + b[i];

}

// vector subtract
template <typename DataTy = float>
inline void vsub(const std::vector<DataTy> &in_a, const std::vector<DataTy> &in_b, std::vector<DataTy> &out) {
	for (size_t i = 0; i < out.size(); ++i) out[i] = in_a[i] - in_b[i];
}

// vector multiply
template <typename DataTy = float>
inline void vmul(const std::vector<DataTy> &in_a, const std::vector<DataTy> &in_b, std::vector<DataTy> &out) {
	for (size_t i = 0; i < out.size(); ++i) out[i] = in_a[i] * in_b[i];
}

// vector divide
template <typename DataTy = float>
inline void vdiv(const std::vector<DataTy> &in_a, const std::vector<DataTy> &in_b, std::vector<DataTy> &out) {
	for (size_t i = 0; i < out.size(); ++i) {
		assert(in_b[i] != 0);
		out[i] = in_a[i] / in_b[i];
	}
}

// vector add scalar
template <typename DataTy = float>
inline void add_scalar(const DataTy alpha, std::vector<DataTy> Y) {
	for (size_t i = 0; i < Y.size(); ++i) Y[i] += alpha;
}

// vector subtract scalar
template <typename DataTy = float>
inline void sub_scalar(const DataTy alpha, std::vector<DataTy> Y) {
	for (size_t i = 0; i < Y.size(); ++i) Y[i] -= alpha;
}

// vector multiply scalar
template <typename DataTy = float>
inline void mul_scalar(const DataTy alpha, std::vector<DataTy> Y) {
	for (size_t i = 0; i < Y.size(); ++i) Y[i] *= alpha;
}

// vector divide scalar
template <typename DataTy = float>
inline void div_scalar(const DataTy alpha, std::vector<DataTy> Y) {
	assert(alpha != 0);
	for (size_t i = 0; i < Y.size(); ++i) Y[i] /= alpha;
}

// dot product
template <typename DataTy = float>
inline DataTy dot(const std::vector<DataTy> x, const std::vector<DataTy> &y) {
	DataTy sum = 0;
	for (size_t i = 0; i < x.size(); ++i)
		sum += x[i] * y[i];
	return sum;
}

// matrix-vector multiply
inline void mvmul(const vec_t &matrix, const vec_t &in_vector, vec_t &out_vector) {
	size_t m = out_vector.size();
	size_t n = in_vector.size();
	for (size_t i = 0; i < m; ++i) { 
		for (size_t j = 0; j < n; ++j) { 
			out_vector[i] += matrix[i*n+j] * in_vector[j];
		} 
	} 
}

// vector-vector multiply
inline void vvmul(const vec_t &a, const vec_t &b, tensor_t &out) {
	size_t m = a.size();
	size_t n = b.size();
	for (size_t i = 0; i < m; ++i) { 
		for (size_t j = 0; j < n; ++j) { 
			out[i][j] += a[i] * b[j];
		} 
	} 
}

// matrix addition
inline void matadd(size_t x, size_t y, const tensor_t &A, const tensor_t &B, tensor_t &C) {
	for (size_t i = 0; i < x; ++i)
		for (size_t j = 0; j < y; ++j)
			C[i][j] = A[i][j] + B[i][j];
}

// matrix multiply: all 2D
inline void matmul2D(const tensor_t &A, const tensor_t &B, tensor_t &C) {
	// A: x*z; B: z*y; C: x*y
	size_t dim_x = A.size();
	size_t dim_y = C[0].size();
	size_t dim_z = A[0].size();
	assert(C.size() == dim_x);
	assert(B.size() == dim_z);
	assert(B[0].size() == dim_y);

	for (size_t i = 0; i < dim_x; ++i) { 
		for (size_t j = 0; j < dim_y; ++j) { 
			C[i][j] = 0;
			for (size_t k = 0; k < dim_z; ++k) { 
				C[i][j] += A[i][k] * B[k][j];
			} 
		} 
	} 
}

inline void matmul1D1D(const size_t dim_x, const size_t dim_y, const size_t dim_z, 
	const vec_t &A, const vec_t &B, vec_t &C) {
#ifdef WITH_BLAS
	const int M = dim_x;
	const int N = dim_y;
	const int K = dim_z;
	const float alpha = 1.0;
	const float beta = 0.0;
	const CBLAS_TRANSPOSE TransA = CblasNoTrans;
	const CBLAS_TRANSPOSE TransB = CblasNoTrans;
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;
	cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, &A[0], lda, &B[0], ldb, beta, &C[0], N);
#else
	//std::cout << "using naive matmul, slow\n";
	assert(A.size() == dim_x*dim_z);
	assert(B.size() == dim_z*dim_y);
	assert(C.size() == dim_x*dim_y);

	for (size_t i = 0; i < dim_x; ++i) { 
		for (size_t j = 0; j < dim_y; ++j) { 
			C[i*dim_y+j] = 0;
			for (size_t k = 0; k < dim_z; ++k) { 
				C[i*dim_y+j] += A[i*dim_z+k] * B[k*dim_y+j];
			} 
		} 
	} 
#endif
}

inline void matmul2D1D(const tensor_t &A, const tensor_t &B, vec_t &C) {
	// A: x*z; B: z*y; C: x*y
	size_t dim_x = A.size();
	size_t dim_y = B[0].size();
	size_t dim_z = A[0].size();
	assert(C.size() == dim_x*dim_y);
	assert(B.size() == dim_z);

#ifdef WITH_BLAS
	vec_t A1D(dim_x*dim_z);
	vec_t B1D(dim_z*dim_y);
	auto ptr = &A1D[0];
	for (size_t i = 0; i < dim_x; i++) {
		std::copy(A[i].begin(), A[i].end(), ptr);
		ptr += dim_z;
	}
	ptr = &B1D[0];
	for (size_t i = 0; i < dim_z; i++) {
		std::copy(B[i].begin(), B[i].end(), ptr);
		ptr += dim_y;
	}
	matmul1D1D(dim_x, dim_y, dim_z, A1D, B1D, C);
#else
	for (size_t i = 0; i < dim_x; ++i) { 
		for (size_t j = 0; j < dim_y; ++j) { 
			C[i*dim_y+j] = 0;
			for (size_t k = 0; k < dim_z; ++k) { 
				C[i*dim_y+j] += A[i][k] * B[k][j];
			} 
		} 
	} 
#endif
}

// matrix multiply
inline void matmul(const tensor_t &A, const vec_t &B, tensor_t &C) {
	// A: x*z; B: z*y; C: x*y
	size_t dim_x = C.size();
	size_t dim_y = C[0].size();
	size_t dim_z = A[0].size();
	assert(A.size() == dim_x);
	assert(B.size() == dim_y*dim_z);

#ifdef WITH_BLAS
	vec_t A1D(dim_x*dim_z);
	vec_t C1D(dim_x*dim_y, 0);
	auto ptr = &A1D[0];
	for (size_t i = 0; i < dim_x; i++) {
		std::copy(A[i].begin(), A[i].end(), ptr);
		ptr += dim_z;
	}
	matmul1D1D(dim_x, dim_y, dim_z, A1D, B, C1D);
	for (size_t i = 0; i < dim_x; i++) {
		for (size_t j = 0; j < dim_y; ++j) { 
			C[i][j] = C1D[i*dim_y+j];
		}
	}
#else
	for (size_t i = 0; i < dim_x; ++i) { 
		for (size_t j = 0; j < dim_y; ++j) { 
			C[i][j] = 0;
			for (size_t k = 0; k < dim_z; ++k) { 
				C[i][j] += A[i][k] * B[k*dim_y+j];
			} 
		} 
	} 
#endif
}

template <typename DataTy = float>
inline void transpose2D(const tensor_t &in, tensor_t &out) {
	size_t x = in.size();
	size_t y = in[0].size();
	for (size_t i = 0; i < y; i ++) {
		for (size_t j = 0; j < x; j ++) {
			out[i][j] = in[j][i];
		}
	}
}

template <typename DataTy = float>
inline void transpose(size_t x, size_t y, const vec_t &in, vec_t &out) {
	for (size_t i = 0; i < y; i ++) {
		for (size_t j = 0; j < x; j ++) {
			out[i*x+j] = in[j*y+i];
		}
	}
}

template <typename DataTy = float>
inline int argmax(const size_t n, const std::vector<DataTy> &x) {
	DataTy max = x[0];
	int max_ind = 0;
	for (size_t i = 1; i < n; i++) {
		if (x[i] > max) {
			max_ind = i;
			max = x[i];
		}
	}
	return max_ind;
}

inline void clear(vec_t &in) {
	for (size_t i = 0; i < in.size(); i++) in[i] = 0;
}

inline void update_all(Graph *g, const tensor_t &in, tensor_t &out, bool norm, const vec_t norm_factor) {
	galois::do_all(galois::iterate(g->begin(), g->end()), [&](const auto& src) {
		clear(out[src]);
		float_t a = 0.0, b = 0.0;
		if (norm) a = norm_factor[src];
		// gather neighbors' embeddings
		for (const auto e : g->edges(src)) {
			const auto dst = g->getEdgeDst(e);
			if (norm) {
				b = a * norm_factor[dst];
				vec_t neighbor = in[dst];
				mul_scalar(b, neighbor);
				vadd(out[src], neighbor, out[src]); // out[src] += in[dst]
			} else vadd(out[src], in[dst], out[src]); // out[src] += in[dst]
		}
	}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("update_all"));
}

template <typename DataTy = float>
inline void relu(const std::vector<DataTy> &in, std::vector<DataTy> &out) {
	for (size_t i = 0; i < out.size(); ++i) {
		out[i] = std::max(in[i], (DataTy)0) + negative_slope * std::min(in[i], (DataTy)0);
	}
}

template <typename DataTy = float>
inline void d_relu(const std::vector<DataTy> &in_diff, const std::vector<DataTy> &fv, std::vector<DataTy> &out_diff) {
	for (size_t i = 0; i < out_diff.size(); ++i) {
		out_diff[i] = in_diff[i] * ((fv[i] > (DataTy)0)  + negative_slope * (fv[i] <= (DataTy)0));
	}
}

inline void d_mvmul(vec_t &in_diff, vec_t &h_in, tensor_t &out_diff) {
	vvmul(h_in, in_diff, out_diff); // transposed feature matrix X^T times in_diff 
}

inline void d_vadd(vec_t &in_diff, vec_t &out_diff) {
	for (size_t i = 0; i < out_diff.size(); ++i)
		out_diff[i] = in_diff[i];
}

template <typename DataTy = float>
inline float reduce_mean(const std::vector<DataTy> &x) {
	size_t n = x.size();
	assert(n > 0);
	float sum = (float)x[0];
	for (size_t i = 1; i < n; i++) {
		sum += (float)x[i];
	}
	return sum / (float)n;
}

const float scale_ = 1. / (1. - dropout_rate);
inline void dropout(const vec_t &in, std::vector<unsigned> &mask, vec_t &out) {
	assert(mask.size() == out.size());
	//rng_bernoulli(1. - dropout_rate, mask); // Create random numbers
	for (size_t i = 0; i < in.size(); ++i)
		mask[i] = bernoulli(dropout_rate);
	for (size_t i = 0; i < in.size(); ++i)
		out[i] = in[i] * mask[i] * scale_;
}

inline void d_dropout(const vec_t &in_diff, std::vector<unsigned> &mask, vec_t &out_diff) {
	for (size_t i = 0; i < in_diff.size(); ++i)
		out_diff[i] = in_diff[i] * mask[i] * scale_;
}

template <typename DataTy = float>
inline DataTy sigmoid_func(DataTy x) {
	return 0.5 * tanh(0.5 * x) + 0.5;
}

// Sigmoid
template <typename DataTy = float>
inline void sigmoid(std::vector<DataTy> &fv) {
	size_t count = fv.size();
	for (size_t i = 0; i < count; ++i) {
		fv[i] = sigmoid_func(fv[i]);
	}
}

// Softmax function takes an N-dimensional vector (X) of real number,
// and transforms it into a vector of real number in range (0,1) which add upto 1.
// To make softmax func numerically stable, we simply normalize the values in the vector, 
// by multiplying the numerator and denominator with a constant C, where log(C)=-max(X)
//    exps = np.exp(X - np.max(X))
//    exps / np.sum(exps)
template <typename DataTy = float>
inline void softmax(const std::vector<DataTy> &input, std::vector<DataTy> &output) {
	const float_t max = *std::max_element(input.begin(), input.end());
	float_t denominator(0);
	for (size_t i = 0; i < input.size(); i++) {
		output[i] = std::exp(input[i] - max);
		denominator += output[i];
	}
	for (size_t i = 0; i < input.size(); i++)
		output[i] /= denominator;
}

template <typename DataTy = float>
inline void log_softmax(const std::vector<DataTy> &input, std::vector<DataTy> &output) {
	const float_t max = *std::max_element(input.begin(), input.end());
	float_t denominator(0);
	for (size_t i = 0; i < input.size(); i++)
		denominator += std::exp(input[i] - max);
	for (size_t i = 0; i < input.size(); i++)
		output[i] = input[i] - max - denominator;
}

// Due to the desirable property of softmax function outputting a probability distribution, 
// we often use it as the final layer in neural networks.
// For this we need to calculate the derivative or gradient,
// and pass it back to the previous layer during backpropagation.
template <typename DataTy = float>
inline void d_softmax(const std::vector<DataTy> &y, const std::vector<DataTy> &p, 
		std::vector<DataTy> &dy, const std::vector<DataTy> &dp) {
	auto n = y.size();
	vec_t df(n, 0);
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			//DataTy delta_ij = i == j? 1 : 0;
			//df[i] += p[j] * (delta_ij - p[i]);
			df[j] = (j == i) ? p[i] * (float_t(1) - p[i]) : -p[j] * p[i];
		}
		// dy = dp * (gradient of softmax)
		dy[i] = dot(dp, df);
	}
/* 
	for (size_t j = 0; j < x.size(); j++) {
		for (size_t k = 0; k < x.size(); k++) {
			df[k] = (k == j) ? y[j] * (float_t(1) - y[j]) : -y[k] * y[j];
		}
		dx[j] = vectorize::dot(&dy[0], &df[0], len);
	}
*/
}

// cross-entropy loss function for multi-class classification
// y: ground truth
// p: predicted probability
template <typename DataTy = float>
inline DataTy cross_entropy(const std::vector<DataTy> &y, const std::vector<DataTy> &p) {
	auto n = y.size();
	assert(n > 0);
	DataTy loss = 0.0;
	for (size_t i = 0; i < n; i++) {
		if (y[i] == float_t(0)) continue;
		if (p[i] == float_t(0)) loss -= y[i] * std::log(float_t(1e-10));
		//if (p[i]==float_t(1)) loss -= (float_t(1) - y[i]) * std::log(float_t(1e-10));
		else loss -= y[i] * std::log(p[i]);// + (float_t(1) - y[i]) * std::log(float_t(1) - p[i]);
		//loss -= y[i] * std::log(p[i]);
	}
	return loss;
}

template <typename DataTy = float>
inline void d_cross_entropy(const std::vector<DataTy> &y, const std::vector<DataTy> &p, std::vector<DataTy> &d) {
	auto n = y.size();
	//for (size_t i = 0; i < n; i++) d[i] = (p[i] - y[i]) / (p[i] * (float_t(1) - p[i]));
	for (size_t i = 0; i < n; i++) {
		d[i] = -y[i] / (p[i] + float_t(1e-10));
		//d[i] = p[i] - y[i];
	}
}

#endif
