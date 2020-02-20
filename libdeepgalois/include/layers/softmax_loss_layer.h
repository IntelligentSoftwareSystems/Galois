#pragma once
#include "layer.h"

class softmax_loss_layer: public layer {
public:
	softmax_loss_layer(unsigned level, std::vector<size_t> in_dims, 
		std::vector<size_t> out_dims, LabelList *lab);
	softmax_loss_layer(unsigned level, std::vector<size_t> in_dims, 
		std::vector<size_t> out_dims) : softmax_loss_layer(level, in_dims, out_dims, NULL) {}
	~softmax_loss_layer() {}
	std::string layer_type() const override { return std::string("softmax_loss"); }
	virtual void forward_propagation(const tensor_t &in_data, tensor_t &out_data);
	virtual void back_propagation(const tensor_t &in_data, const tensor_t &out_data, tensor_t &out_grad, tensor_t &in_grad);

private:
	LabelList *labels;
};

