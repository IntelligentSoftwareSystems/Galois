#pragma once
#include "layer.h"

// ReLU Layer
class relu_layer : public layer {
public:
	relu_layer(unsigned level, std::vector<size_t> in_dims, std::vector<size_t> out_dims)
		: layer(level, in_dims, out_dims) {
		trainable_ = false;
	}

	// fv: input feature vectors (tensor)
	inline void forward(tensor_t &fv) {
		size_t n = fv.size(); // num_samples
		if (n == 0) return;
		size_t dim = fv[0].size(); // feature dimension
		galois::do_all(galois::iterate((size_t)0, n), [&](const auto& i) {
			for (size_t i = 0; i < dim; ++i) 
				fv[i] = std::max(fv[i], 0.0) + negative_slope * std::min(fv[i], 0.0);
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("relu_layer-fw"));
	}

};
