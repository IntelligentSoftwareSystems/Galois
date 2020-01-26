#ifndef _MATH_FUNCTIONS_
#define _MATH_FUNCTIONS_
#include <cmath>

// vector add
void vadd(size_t dim, FV in_a, FV in_b, FV &out) {
	for (size_t i = 0; i < dim; ++i) {
		out[i] = in_a[i] + in_b[i];
	}
}

// vector subtract
void vsub(size_t dim, FV in_a, FV in_b, FV &out) {
	for (size_t i = 0; i < dim; ++i) {
		out[i] = in_a[i] - in_b[i];
	}
}

// vector multiply
void vmul(size_t dim, FV in_a, FV in_b, FV &out) {
	for (size_t i = 0; i < dim; ++i) {
		out[i] = in_a[i] * in_b[i];
	}
}

// vector divide
void vdiv(size_t dim, FV in_a, FV in_b, FV &out) {
	for (size_t i = 0; i < dim; ++i) {
		out[i] = in_a[i] / in_b[i];
	}
}

// dot product
FeatureT dot(const size_t n, const FV x, const FV y) {
	FeatureT sum = 0;
	for (size_t i = 0; i < n; ++i) {
		sum += x[i] * y[i];
	}
	return sum;
}

// matrix-vector multiply
void mvmul(size_t dim, FV2D matrix, FV in_vector, FV &out_vector) {
	for (size_t i = 0; i < dim; ++i) { 
		for (size_t j = 0; j < dim; ++j) { 
			out_vector[i] += matrix[i][j] * in_vector[j];
		} 
	} 
}

// matrix multiply
void matmul(size_t dim, FV2D A, FV2D B, FV2D &C) {
	for (size_t i = 0; i < dim; ++i) { 
		for (size_t j = 0; j < dim; ++j) { 
			for (size_t k = 0; k < dim; ++k) { 
				C[i][j] += A[i][k] * B[k][j];
			} 
		} 
	} 
}

int argmax(const size_t n, const FV x) {
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
float reduce_mean(const std::vector<DataTy> x) {
	size_t n = x.size();
	float sum = (float)x[0];
	for (size_t i = 1; i < n; i++) {
		sum += (float)x[i];
	}
	return sum / (float)n;
}

// ReLU
const float negative_slope = 0;
void relu(FV &fv) {
	size_t count = fv.size();
	for (size_t i = 0; i < count; ++i) {
		fv[i] = std::max(fv[i], (FeatureT)0) + negative_slope * std::min(fv[i], (FeatureT)0);
	}
}

inline FeatureT sigmoid_func(FeatureT x) {
	return 0.5 * tanh(0.5 * x) + 0.5;
}

// Sigmoid
void sigmoid(FV &fv) {
	size_t count = fv.size();
	for (size_t i = 0; i < count; ++i) {
		fv[i] = sigmoid_func(fv[i]);
	}
}

//Softmax
void softmax(FV2D features) {
}

void softmax_cross_entropy_with_logits(LabelList preds, LabelList labels, std::vector<float> &loss) {
}
#endif
