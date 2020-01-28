#ifndef _MATH_FUNCTIONS_
#define _MATH_FUNCTIONS_
#include <cmath>
#include "random.h"

// vector add
inline void vadd(const FV &in_a, const FV &in_b, FV &out) {
	size_t dim = out.size();
	for (size_t i = 0; i < dim; ++i) {
		out[i] = in_a[i] + in_b[i];
	}
}

// vector subtract
inline void vsub(const FV &in_a, const FV &in_b, FV &out) {
	size_t dim = out.size();
	for (size_t i = 0; i < dim; ++i) {
		out[i] = in_a[i] - in_b[i];
	}
}

// vector multiply
inline void vmul(const FV &in_a, const FV &in_b, FV &out) {
	size_t dim = out.size();
	for (size_t i = 0; i < dim; ++i) {
		out[i] = in_a[i] * in_b[i];
	}
}

// vector divide
inline void vdiv(size_t dim, const FV &in_a, const FV &in_b, FV &out) {
	for (size_t i = 0; i < dim; ++i) {
		out[i] = in_a[i] / in_b[i];
	}
}

// dot product
inline FeatureT dot(const size_t n, const FV &x, const FV &y) {
	FeatureT sum = 0;
	for (size_t i = 0; i < n; ++i) {
		sum += x[i] * y[i];
	}
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

// ReLU
const float negative_slope = 0;
inline void relu(FV &fv) {
	size_t count = fv.size();
	for (size_t i = 0; i < count; ++i) {
		fv[i] = std::max(fv[i], (FeatureT)0) + negative_slope * std::min(fv[i], (FeatureT)0);
	}
}

inline void d_relu(FV &in_diff, FV &fv, FV &out_diff) {
	size_t count = out_diff.size();
	for (size_t i = 0; i < count; ++i) {
		out_diff[i] = in_diff[i] * ((fv[i] > (FeatureT)0)  + negative_slope * (fv[i] <= (FeatureT)0));
	}
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

//Softmax
template <typename DataTy = float>
inline void softmax(std::vector<DataTy> &input, std::vector<DataTy> &output) {
	auto n = input.size();
	//DataTy m = *(std::max_element(input.begin(), input.end()));
	DataTy m = -INFINITY;
	for (size_t i = 0; i < n; i++) if (input[i] > m) m = input[i];
	DataTy sum = (DataTy)0;
	for (size_t i = 0; i < n; i++) sum += expf(input[i]-m);
	DataTy offset = m + logf(sum);
	for (size_t i = 0; i < n; i++) output[i] = expf(input[i]-offset);
}

template <typename DataTy = float>
inline DataTy cross_entropy(std::vector<DataTy> &y, std::vector<DataTy> &p) {
	auto n = y.size();
	DataTy loss = 0.0;
	for (size_t i = 0; i < n; i++) loss -= y[i] * logf(p[i]);
	return loss / (DataTy)n;
}

// TODO: need optimization
inline void softmax_cross_entropy_with_logits(FV2D &h_out, LabelList &labels, std::vector<float> &loss) {
	auto n = h_out.size(); // V
	auto m = h_out[0].size(); // E
	std::vector<float> y(m); // ground truth
	std::vector<float> p(m); // prediction
	for (size_t i = 0; i < n; i++) {
		if (labels[i] < 0) continue; // masked
		softmax(h_out[i], p);
		for (size_t j = 0; j < m; j ++) y[j] = 0.0; // ground truth
		assert(labels[i] < m);
		y[labels[i]] = 1.0;
		loss[i] = cross_entropy(y, p);
	}
}
#endif
