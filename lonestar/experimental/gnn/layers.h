#ifndef _LAYERS_H_
#define _LAYERS_H_
#include <cmath>

// Conv Layer
inline void d_mvmul(FV &in_diff, FV &out_diff) {
	for (size_t i = 0; i < out_diff.size(); ++i)
		out_diff[i] = in_diff[i];
}
inline void d_vadd(FV &in_diff, FV &out_diff) {
	for (size_t i = 0; i < out_diff.size(); ++i)
		out_diff[i] = in_diff[i];
}

// ReLU Layer
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

// Softmax Layer
// Softmax function takes an N-dimensional vector (X) of real number,
// and transforms it into a vector of real number in range (0,1) which add upto 1.
// To make softmax func numerically stable, we simply normalize the values in the vector, 
// by multiplying the numerator and denominator with a constant C, where log(C)=-max(X)
//    exps = np.exp(X - np.max(X))
//    exps / np.sum(exps)
template <typename DataTy = float>
inline void softmax(std::vector<DataTy> &input, std::vector<DataTy> &output) {
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
inline void d_cross_entropy(std::vector<DataTy> &y, std::vector<DataTy> &p, std::vector<DataTy> &out_diff) {
	auto n = y.size();
	//softmax(x, out_diff);
	for (size_t i = 0; i < n; i++) out_diff[i] = p[i] - y[i];
}

#endif
