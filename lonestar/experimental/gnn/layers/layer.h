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
#include "../node.h"
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

class layer : public node {
public:
	layer(unsigned level, std::vector<size_t> in_dims, std::vector<size_t> out_dims) :
		node(in_dims.size(), out_dims.size()),
		act_(false), level_(level), num_dims(in_dims.size()),
		input_dims(in_dims), output_dims(out_dims) { add_edge(); }
	virtual ~layer() = default;
	virtual void forward_propagation(const tensor_t &in_data, tensor_t &out_data) = 0;
	virtual void back_propagation(const tensor_t &in_data, const tensor_t &out_data,
			tensor_t &out_grad, tensor_t &in_grad) = 0;
	virtual std::string layer_type() const = 0;
	//virtual void setup(Graph *g, vec_t *diff, LabelList *lab) = 0;

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
	void set_in_data(tensor_t data) {
		prev_ = std::make_shared<edge>(this, input_dims[1]);
		prev_->get_data() = data;
		prev_->get_gradient().resize(input_dims[0]);
		// allocate memory for intermediate gradients
		std::cout << "l0 in_grad alloc: x=" << output_dims[0] << ", y=" << output_dims[1] << "\n";
		for (size_t i = 0; i < input_dims[0]; ++i)
			prev_->get_gradient()[i].resize(input_dims[1]);
	}
	void add_edge() {
		// add an outgoing edge
		next_ = std::make_shared<edge>(this, output_dims[1]);
		// allocate memory for intermediate feature vectors
		next_->get_data().resize(output_dims[0]);
		for (size_t i = 0; i < output_dims[0]; ++i)
			next_->get_data()[i].resize(output_dims[1]);
	}
	void alloc_grad() {
		// allocate memory for intermediate gradients
		std::cout << "l" << level_ << " out_grad alloc: x=" << output_dims[0] << ", y=" << output_dims[1] << "\n";
		next_->get_gradient().resize(output_dims[0]);
		for (size_t i = 0; i < output_dims[0]; ++i)
			next_->get_gradient()[i].resize(output_dims[1]);
	}
	void forward() {
		forward_propagation(prev()->get_data(), next()->get_data());
	}
	void backward() {
		back_propagation(prev()->get_data(), next()->get_data(), next()->get_gradient(), prev()->get_gradient());
	}
	void update_weight(optimizer *opt) {
		// parallelize only when target size is big enough to mitigate thread spawning overhead.
		bool parallel = (W.size() >= 512);
		vec_t diff;
		prev()->merge_grads(&diff);
		auto in_data = prev()->get_data();
		float_t rcp_batch_size = float_t(1.0) / in_data.size();
		for (size_t i = 0; i < diff.size(); ++i)
			diff[i] *= rcp_batch_size;
		opt->update(diff, W, parallel); // W += diff
		prev()->clear_grads();
	}
	inline acc_t get_masked_loss(MaskList &masks) {
		size_t n = loss.size();
		assert(n > 0);
		acc_t sum_mask = std::accumulate(masks.begin(), masks.end(), (acc_t)0);
		acc_t avg_mask = sum_mask / (acc_t)n;
		for (size_t i = 0; i < n; i ++) loss[i] = loss[i] * (acc_t)(masks[i]) / avg_mask;
		acc_t sum_loss = std::accumulate(loss.begin(), loss.end(), (acc_t)0);
		return sum_loss / (acc_t)n;
	}

protected:
	bool act_;
	unsigned level_;
	size_t num_dims;
	std::vector<size_t> input_dims;
	std::vector<size_t> output_dims;
	std::string name_;
	bool trainable_;
	vec_t W; // parameters to learn, for vertex v, layer0: D x 16, layer1: 16 x E
	vec_t Q; // parameters to learn, for vertex u, i.e. v's neighbors, layer0: D x 16, layer1: 16 x E
	vec_t loss; // error for each vertex: N x 1
};

// head: layer i+1, tail: layer i
inline void connect(layer *head, layer *tail,
    	size_t head_index = 0, size_t tail_index = 0) {
	//auto out_shape = head->out_shape()[head_index];
	//auto in_shape  = tail->in_shape()[tail_index];
	//head->setup(false);
	//if (in_shape.size() == 0) {
	//	tail->set_in_shape(out_shape);
	//	in_shape = out_shape;
	//}
	//if (out_shape.size() != in_shape.size()) 
	//	connection_mismatch(*head, *tail);
	//if (!head->next_[head_index])
	//	throw nn_error("output edge must not be null");
	tail->prev_ = head->next_;
	tail->prev_->add_next_node(tail);
}

