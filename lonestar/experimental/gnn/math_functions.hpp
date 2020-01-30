#ifndef _MATH_FUNCTIONS_
#define _MATH_FUNCTIONS_
#include <cmath>
#include "random.h"
#include <immintrin.h>

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
	for (size_t i = 0; i < out.size(); ++i) out[i] = in_a[i] / in_b[i];
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
inline void mvmul(const FV2D &matrix, const FV &in_vector, FV &out_vector) {
	size_t m = out_vector.size();
	size_t n = in_vector.size();
	for (size_t i = 0; i < m; ++i) { 
		for (size_t j = 0; j < n; ++j) { 
			out_vector[i] += matrix[i][j] * in_vector[j];
		} 
	} 
}

// vector-vector multiply
inline void vvmul(const FV &a, const FV &b, FV2D &out) {
	size_t m = a.size();
	size_t n = b.size();
	for (size_t i = 0; i < m; ++i) { 
		for (size_t j = 0; j < n; ++j) { 
			out[i][j] += a[i] * b[j];
		} 
	} 
}

// matrix addition
inline void matadd(size_t x, size_t y, const FV2D &A, const FV2D &B, FV2D &C) {
	for (size_t i = 0; i < x; ++i)
		for (size_t j = 0; j < y; ++j)
			C[i][j] = A[i][j] + B[i][j];
}

// matrix multiply
inline void matmul(size_t dim, const FV2D &A, const FV2D &B, FV2D &C) {
	for (size_t i = 0; i < dim; ++i) { 
		for (size_t j = 0; j < dim; ++j) { 
			for (size_t k = 0; k < dim; ++k) { 
				C[i][j] += A[i][k] * B[k][j];
			} 
		} 
	} 
}

inline int argmax(const size_t n, const FV &x) {
	FeatureT max = x[0];
	int max_ind = 0;
	for (size_t i = 1; i < n; i++) {
		if (x[i] > max) {
			max_ind = i;
			max = x[i];
		}
	}
	return max_ind;
}

template <typename DataTy = float>
inline float reduce_mean(const std::vector<DataTy> &x) {
	size_t n = x.size();
	float sum = (float)x[0];
	for (size_t i = 1; i < n; i++) {
		sum += (float)x[i];
	}
	return sum / (float)n;
}

#include <boost/random/bernoulli_distribution.hpp>
template <typename Dtype=float>
void rng_bernoulli(const int n, const Dtype p, std::vector<unsigned> r) {
	boost::bernoulli_distribution<Dtype> random_distribution(p);
	boost::variate_generator<rng_t*, boost::bernoulli_distribution<Dtype> >
		variate_generator(rng(), random_distribution);
	for (int i = 0; i < n; ++i)
		r[i] = static_cast<unsigned>(variate_generator());
}

inline void dropout(FV &in, std::vector<unsigned> &mask, FV &out) {
	size_t count = in.size();
	float threshold_ = dropout_rate;
	float scale_ = 1. / (1. - threshold_);
	// Create random numbers
	rng_bernoulli(count, 1. - threshold_, mask);
	for (size_t i = 0; i < count; ++i)
		out[i] = in[i] * mask[i] * scale_;
}

inline void d_dropout(FV &in_diff, FV &mask, FV &out_diff) {
	size_t count = in_diff.size();
	float threshold_ = dropout_rate;
	float scale_ = 1. / (1. - threshold_);
	for (size_t i = 0; i < count; ++i)
		out_diff[i] = in_diff[i] * mask[i] * scale_;
}

inline FeatureT sigmoid_func(FeatureT x) {
	return 0.5 * tanh(0.5 * x) + 0.5;
}

// Sigmoid
inline void sigmoid(FV &fv) {
	size_t count = fv.size();
	for (size_t i = 0; i < count; ++i) {
		fv[i] = sigmoid_func(fv[i]);
	}
}

#endif
