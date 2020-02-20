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
#include "../utils.h"
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
		level_(level), begin_(0), end_(0), num_dims(in_dims.size()),
		input_dims(in_dims), output_dims(out_dims) { add_edge(); }
	virtual ~layer() = default;
	virtual void forward_propagation(const tensor_t &in_data, tensor_t &out_data) = 0;
	virtual void back_propagation(const tensor_t &in_data, const tensor_t &out_data,
			tensor_t &out_grad, tensor_t &in_grad) = 0;
	virtual std::string layer_type() const = 0;
	virtual void set_context(net_phase ctx) {}
	//virtual void setup(Graph *g, vec_t *diff, LabelList *lab) = 0;

	void set_trainable(bool trainable) { trainable_ = trainable; }
	bool trainable() const { return trainable_; }
	void set_name(std::string name) { name_ = name; }
	std::string get_name() { return name_; }
	void print_layer_info() {
		std::cout << "Layer" << level_ << " type: " << layer_type()
			<< " input[" << input_dims[0] << "," << input_dims[1] 
			<< "] output[" << output_dims[0] << "," << output_dims[1] << "]\n";
	}
	virtual void set_sample_mask(size_t sample_begin, size_t sample_end, size_t sample_count, MaskList &masks) {
		begin_ = sample_begin;
		end_ = sample_end;
		count_ = sample_count;
		masks_ = masks;
	}
	void set_in_data(tensor_t data) {
		prev_ = std::make_shared<edge>(this, input_dims[1]);
		prev_->get_data() = data;
		prev_->get_gradient().resize(input_dims[0]);
		// allocate memory for intermediate gradients
		//std::cout << "l0 in_grad alloc: x=" << output_dims[0] << ", y=" << output_dims[1] << "\n";
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
		//std::cout << "l" << level_ << " out_grad alloc: x=" << output_dims[0] << ", y=" << output_dims[1] << "\n";
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
		//std::cout << "[debug] " << name_ << ": updating weight...\n"; 
		// parallelize only when target size is big enough to mitigate thread spawning overhead.
		bool parallel = (W.size() >= 512);
		//vec_t diff;
		//prev()->merge_grads(&diff);
		//auto in_data = prev()->get_data();
		//float_t rcp_batch_size = float_t(1.0) / in_data.size();
		//for (size_t i = 0; i < diff.size(); ++i)
		//	diff[i] *= rcp_batch_size;
		opt->update(weight_grad, W, parallel); // W += grad
		prev()->clear_grads();
	}
	inline acc_t get_masked_loss() {
		//acc_t total_loss = acc_t(0);
		//size_t valid_sample_count = 0;
		AccumF total_loss;
		AccumU valid_sample_count;
		total_loss.reset();
		valid_sample_count.reset();
		//for (size_t i = begin_; i < end_; i ++) {
		galois::do_all(galois::iterate(begin_, end_), [&](const auto& i) {
			if (masks_[i]) {
				total_loss += loss[i];
				valid_sample_count += 1;
			}
		}, galois::chunk_size<256>(), galois::steal(), galois::loopname("getMaskedLoss"));
		//}
		assert(valid_sample_count.reduce() == count_);
		return total_loss.reduce() / (acc_t)count_;
	}

protected:
	unsigned level_; // layer id: [0, num_layers-1]
	size_t begin_; // sample begin index
	size_t end_; // sample end index
	size_t count_; // number of samples
	MaskList masks_; // masks to show which samples are valid
	size_t num_dims; // number of dimensions
	std::vector<size_t> input_dims; // input dimensions
	std::vector<size_t> output_dims; // output dimentions
	std::string name_; // name of this layer
	bool trainable_; // is this layer trainable
	vec_t W; // parameters to learn, for vertex v, layer0: D x 16, layer1: 16 x E
	vec_t Q; // parameters to learn, for vertex u, i.e. v's neighbors, layer0: D x 16, layer1: 16 x E
	vec_t weight_grad; // weight gradient for updating parameters
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

