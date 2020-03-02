#pragma once
/**
 * Code based on below link.
 *
 * https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/layer.h
 *
 * Copyright (c) 2013, Taiga Nomi and the respective contributors
 * All rights reserved.
 * Reused/revised under 3-BSD
 */

#include <queue>
#include <cmath>
#include <vector>
#include <limits>
#include <memory>
#include <string>
#include <cassert>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <utility>
#include <algorithm>
#include <unordered_set>
#include "../node.h"
#include "../types.h"
#include "../utils.h"
#include "../gtypes.h"
#include "deepgalois/context.h"
#include "../optimizer.h"
#include "../math_functions.hh"
/**
 * base class of all kind of NN layers
 *
 * sub-class should override these methods:
 * - forward_propagation ... body of forward-pass calculation
 * - back_propagation    ... body of backward-pass calculation
 * - in_shape            ... specify input data shapes
 * - out_shape           ... specify output data shapes
 * - layer_type          ... name of layer
 *
 * Node inheritance is just to get accessed to linked-list semantics it
 * provides
 **/
class layer : public node {
public:
  layer(unsigned level, std::vector<size_t> in_dims,
        std::vector<size_t> out_dims)
      : node(in_dims.size(), out_dims.size()), level_(level), begin_(0),
        end_(0), num_dims(in_dims.size()), input_dims(in_dims),
        output_dims(out_dims) {
    add_edge();
  }
  virtual ~layer()                       = default;
  virtual std::string layer_type() const = 0;
  virtual void set_netphase(net_phase phase) {}
  //! save context
  virtual void set_context(Context* ctx) { context = ctx; }
  virtual acc_t get_masked_loss() { return acc_t(0); }

  // main functions for layer work
  virtual void forward_propagation(const float_t* in_data,
                                   float_t* out_data)                = 0;
  virtual void back_propagation(const float_t* in_data, const float_t* out_data,
                                float_t* out_grad, float_t* in_grad) = 0;

  // is this layer trainable?
  void set_trainable(bool trainable) { trainable_ = trainable; }
  bool trainable() const { return trainable_; }

  // name metadata
  void set_name(std::string name) { name_ = name; }
  std::string get_name() { return name_; }

  mask_t* get_device_masks() { return d_masks_; }
  //! debug print function
  void print_layer_info() {
    std::cout << "Layer" << level_ << " type: " << layer_type() << " input["
              << input_dims[0] << "," << input_dims[1] << "] output["
              << output_dims[0] << "," << output_dims[1] << "]\n";
  }
  virtual void set_sample_mask(size_t sample_begin, size_t sample_end,
                               size_t sample_count, mask_t* masks) {
    begin_ = sample_begin;
    end_   = sample_end;
    count_ = sample_count;
    masks_ = masks;
#ifndef CPU_ONLY
    copy_masks_device(input_dims[0], masks_, d_masks_);
#endif
  }

  //! set the data of the previous layer connected to this one
  void set_in_data(float_t* data) {
    prev_ = std::make_shared<edge>(this, input_dims[0], input_dims[1]);
    prev_->set_data(data);
    // no need to allocate memory for gradients, since this is the input layer.
  }

  void add_edge() {
    // add an outgoing edge
    next_ = std::make_shared<edge>(this, output_dims[0], output_dims[1]);
    // allocate memory for intermediate feature vectors and gradients
    next_->alloc();
  }
  void alloc_grad() {
    // allocate memory for intermediate gradients
  }

  //! calls forward propagation using previous layer as input and writes
  //! to next layer as output
  void forward() {
    // std::cout << name_ << ": forwarding ... ";
    forward_propagation(prev()->get_data(), next()->get_data());
  }

  //! calls backward propagation
  void backward() {
    // std::cout << name_ << ": backwarding ... ";
    back_propagation(prev()->get_data(), next()->get_data(),
                     next()->get_gradient(), prev()->get_gradient());
  }

  //! use optimizer to update weights given gradient
  void update_weight(optimizer* opt) {
    // vec_t diff;
    // prev()->merge_grads(&diff);
#ifdef CPU_ONLY
    // std::cout << name_ << ": weight updating ... ";
    // parallelize only when target size is big enough to mitigate thread
    // spawning overhead.
    bool parallel = (W.size() >= 512);
    opt->update(weight_grad, W, parallel); // W += grad
#else
	//std::cout << name_ << ": ";
    opt->update_gpu(input_dims[1]*output_dims[1], d_weight_grad, d_W); // W += grad
#endif
    // prev()->clear_grads();
    next()->clear_grads();
  }

protected:
  unsigned level_;                 // layer id: [0, num_layers-1]
  size_t begin_;                   // sample begin index
  size_t end_;                     // sample end index
  size_t count_;                   // number of samples
  size_t num_dims;                 // number of dimensions
  std::vector<size_t> input_dims;  // input dimensions
  std::vector<size_t> output_dims; // output dimentions
  std::string name_;               // name of this layer
  bool trainable_;                 // is this layer trainable
  vec_t W; // parameters to learn, for vertex v, layer0: D x 16, layer1: 16 x E
  vec_t Q; // parameters to learn, for vertex u, i.e. v's neighbors, layer0: D x
           // 16, layer1: 16 x E
  vec_t weight_grad; // weight gradient for updating parameters
  float_t* d_W;
  float_t* d_weight_grad;
  mask_t* masks_; // masks to show which samples are valid
  mask_t* d_masks_;
  float_t* loss; // error for each vertex: N x 1
  Context* context;
};

// head: layer i+1, tail: layer i
inline void connect(layer* head, layer* tail, size_t head_index = 0,
                    size_t tail_index = 0) {
  // auto out_shape = head->out_shape()[head_index];
  // auto in_shape  = tail->in_shape()[tail_index];
  // head->setup(false);
  // if (in_shape.size() == 0) {
  //	tail->set_in_shape(out_shape);
  //	in_shape = out_shape;
  //}
  // if (out_shape.size() != in_shape.size())
  //	connection_mismatch(*head, *tail);
  // if (!head->next_[head_index])
  //	throw nn_error("output edge must not be null");
  tail->prev_ = head->next_;
  tail->prev_->add_next_node(tail);
}
