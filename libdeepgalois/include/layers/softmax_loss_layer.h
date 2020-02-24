#pragma once
#include "layer.h"

class softmax_loss_layer: public layer {
public:
	softmax_loss_layer(unsigned level, std::vector<size_t> in_dims, std::vector<size_t> out_dims);
	~softmax_loss_layer() {}
	std::string layer_type() const override { return std::string("softmax_loss"); }
	//virtual void forward_propagation(const vec_t &in_data, vec_t &out_data);
	virtual void forward_propagation(const float_t *in_data, float_t *out_data);
	//virtual void back_propagation(const vec_t &in_data, const vec_t &out_data, vec_t &out_grad, vec_t &in_grad);
	virtual void back_propagation(const float_t *in_data, const float_t *out_data, float_t *out_grad, float_t *in_grad);
};

