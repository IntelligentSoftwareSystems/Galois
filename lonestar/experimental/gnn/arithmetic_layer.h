#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "layer.h"

// element-wise add N vectors ```y_i = x0_i + x1_i + ... + xnum_i```
class elementwise_add_layer : public layer {
public:
	// @param num_args [in] number of inputs
	// @param dim      [in] number of elements for each input
	elementwise_add_layer(size_t num_args, size_t dim)
		: layer(std::vector<vector_type>(num_args, vector_type::data), {vector_type::data}),
		num_args_(num_args), dim_(dim) {}

	void forward(FV2D &fv) { }
	void backward(FV2D &fv) { }
};
