#ifndef _MATH_FUNCTIONS_
#define _MATH_FUNCTIONS_
#include <cmath>
#include "random.h"
#include <immintrin.h>
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
inline void matmul(const FV2D &A, const vec_t &B, FV2D &C) {
	size_t dim_x = A.size();
	size_t dim_y = A[0].size();
	size_t dim_z = C[0].size();
	assert(C.size() == dim_x);

	for (size_t i = 0; i < dim_x; ++i) { 
		for (size_t j = 0; j < dim_y; ++j) { 
			for (size_t k = 0; k < dim_z; ++k) { 
				C[i][k] += A[i][j] * B[j*dim_z+k];
			} 
		} 
	} 
}

template <typename DataTy = float>
inline void transpose(const FV2D &in, FV2D &out) {
	size_t x = in.size();
	size_t y = in[0].size();
	for (size_t i = 0; i < y; i ++) {
		for (size_t j = 0; j < x; j ++) {
			out[i][j] = in[j][i];
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

inline void update_all(Graph *g, size_t dim, const FV2D &in, FV2D &out) {
	galois::do_all(galois::iterate(g->begin(), g->end()), [&](const auto& src) {
		out[src].resize(dim, 0); // used to gather neighbors' embeddings
		for (const auto e : g->edges(src)) {
			const auto dst = g->getEdgeDst(e);
			vadd(out[src], in[dst], out[src]); // out[src] += in[dst]
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

inline void d_mvmul(FV &in_diff, FV &h_in, FV2D &out_diff) {
	vvmul(h_in, in_diff, out_diff); // transposed feature matrix X^T times in_diff 
}

inline void d_vadd(FV &in_diff, FV &out_diff) {
	for (size_t i = 0; i < out_diff.size(); ++i)
		out_diff[i] = in_diff[i];
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
template <typename DataTy = float>
void rng_bernoulli(const int n, const DataTy p, std::vector<unsigned> r) {
	boost::bernoulli_distribution<DataTy> random_distribution(p);
	boost::variate_generator<rng_t*, boost::bernoulli_distribution<DataTy> >
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
	auto n = input.size();

	// find maximum element
	//DataTy m = *(std::max_element(input.begin(), input.end()));
	DataTy m = -INFINITY;
	for (size_t i = 0; i < n; i++) if (input[i] > m) m = input[i];
	std::vector<DataTy> exps(n, 0);

	// subtraction and exponentiation
	for (size_t i = 0; i < n; i++) exps[i] = expf(input[i]-m);

	// sum after exp
	DataTy sum = (DataTy)0;
	//for (size_t i = 0; i < n; i++) sum += expf(input[i]-m);
	for (size_t i = 0; i < n; i++) sum += exps[i];
	//DataTy offset = m + logf(sum);

	// division
	//for (size_t i = 0; i < n; i++) output[i] = expf(input[i]-offset);
	for (size_t i = 0; i < n; i++) output[i] = exps[i] / sum;
}

// Due to the desirable property of softmax function outputting a probability distribution, 
// we often use it as the final layer in neural networks.
// For this we need to calculate the derivative or gradient,
// and pass it back to the previous layer during backpropagation.
template <typename DataTy = float>
inline void d_softmax(std::vector<DataTy> &y, std::vector<DataTy> &p, std::vector<DataTy> &out_diff) {
	auto n = y.size();
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			DataTy delta_ij = i == j? 1 : 0;
			out_diff[i] += p[j] * (delta_ij - p[i]);
		}
	}
}

// cross entropy
template <typename DataTy = float>
inline DataTy cross_entropy(std::vector<DataTy> &y, std::vector<DataTy> &p) {
	auto n = y.size();
	DataTy loss = 0.0;
	for (size_t i = 0; i < n; i++) loss -= y[i] * logf(p[i]);
	return loss / (DataTy)n;
}

template <typename DataTy = float>
inline void d_cross_entropy(const std::vector<DataTy> &y, const std::vector<DataTy> &p, std::vector<DataTy> &out_diff) {
	auto n = y.size();
	//softmax(x, out_diff);
	for (size_t i = 0; i < n; i++) out_diff[i] = p[i] - y[i];
}

#endif
