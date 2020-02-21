#pragma once
#include "layer.h"

// ReLU Layer
class relu_layer : public layer {
public:
	relu_layer(unsigned level, std::vector<size_t> in_dims, std::vector<size_t> out_dims)
		: layer(level, in_dims, out_dims) {
		trainable_ = false;
	}
	~relu_layer() {}
	std::string layer_type() const override { return std::string("relu"); }
	virtual void forward_propagation(const tensor_t &in_data, tensor_t &out_data);
	virtual void back_propagation(const tensor_t &in_data, const tensor_t &out_data, tensor_t &out_grad, tensor_t &in_grad);
};
