#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/util/util.h"

// element-wise add N vectors ```y_i = x0_i + x1_i + ... + xnum_i```
class elementwise_add_layer : public layer {
public:
	// @param num_args [in] number of inputs
	// @param dim      [in] number of elements for each input
	elementwise_add_layer(size_t num_args, size_t dim)
		: layer(std::vector<vector_type>(num_args, vector_type::data), {vector_type::data}),
		num_args_(num_args), dim_(dim) {}

	// ReLU Layer
	inline void forward(FV2D &fv) {
		// fv: N x Dim
		size_t n = fv.size();
		if (n == 0) return;
		size_t dim = fv[0].size();
		galois::do_all(galois::iterate((size_t)0, n), [&](const auto& i) {
			for (size_t i = 0; i < dim; ++i) 
				fv[i] = std::max(fv[i], (FeatureT)0) + negative_slope * std::min(fv[i], (FeatureT)0);
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("relu0"));
	}

};
