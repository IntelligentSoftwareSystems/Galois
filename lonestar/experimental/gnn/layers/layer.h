#pragma once

#include <queue>
#include <cmath>
#include <vector>
#include <limits>
#include <memory>
#include <string>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <utility>
#include <algorithm>
#include <unordered_set>
#include "../types.h"
#include "../optimizer.h"
#include "../math_functions.hpp"
/**
 * base class of all kind of NN layers
 *
 * sub-class should override these methods:
 * - forward_propagation ... body of forward-pass calculation
 * - back_propagation    ... body of backward-pass calculation
 * - in_shape            ... specify input data shapes
 * - out_shape           ... specify output data shapes
 * - layer_type          ... name of layer
 **/

class layer {
public:
	layer(unsigned level, std::vector<size_t> in_dims, std::vector<size_t> out_dims) :
		act_(false), level_(level), num_dims(in_dims.size()),
		input_dims(in_dims), output_dims(out_dims) {}
	virtual ~layer() = default;
	virtual void forward(const std::vector<FV> &in_data, std::vector<FV> &out_data) = 0;
	virtual void backward(const std::vector<FV> &in_data, const std::vector<FV> &out_data,
			std::vector<FV> &out_grad, std::vector<FV> &in_grad) = 0;
	virtual std::string layer_type() const = 0;
	virtual void setup(Graph *g, FV *d, LabelList *lab) = 0;
	virtual void update_weights(optimizer *opt) = 0;
	void set_act(bool act) { act_ = act; }
	void set_trainable(bool trainable) { trainable_ = trainable; }
	bool trainable() const { return trainable_; }
	void set_name(std::string name) { name_ = name; }
	std::string get_name() { return name_; }
	void print_layer_info() {
		std::cout << "Layer" << level_ << " type: " << layer_type()
			<< " input[" << input_dims[0] << "," << input_dims[1] 
			<< "] output[" << output_dims[0] << "," << output_dims[1] << "]\n";
	}
protected:
	bool act_;
	unsigned level_;
	size_t num_dims;
	std::vector<size_t> input_dims;
	std::vector<size_t> output_dims;
	std::string name_;
	bool trainable_;
};
