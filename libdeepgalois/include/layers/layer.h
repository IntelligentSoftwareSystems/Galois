#pragma once

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
#include "../context.h"
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
  virtual void set_context(Context* ctx) { context = ctx; }
  // virtual void forward_propagation(const vec_t &in_data, vec_t &out_data) =
  // 0; virtual void back_propagation(const vec_t &in_data, const vec_t
  // &out_data, vec_t &out_grad, vec_t &in_grad) = 0;
  virtual void forward_propagation(const float_t* in_data,
                                   float_t* out_data)                = 0;
  virtual void back_propagation(const float_t* in_data, const float_t* out_data,
                                float_t* out_grad, float_t* in_grad) = 0;

  void set_trainable(bool trainable) { trainable_ = trainable; }
  bool trainable() const { return trainable_; }
  void set_name(std::string name) { name_ = name; }
  std::string get_name() { return name_; }
  mask_t* get_device_masks() { return d_masks_; }
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
  void forward() {
    // std::cout << name_ << ": forwarding ... ";
    forward_propagation(prev()->get_data(), next()->get_data());
  }
  void backward() {
    // std::cout << name_ << ": backwarding ... ";
    back_propagation(prev()->get_data(), next()->get_data(),
                     next()->get_gradient(), prev()->get_gradient());
  }
  void update_weight(optimizer* opt) {
    // std::cout << name_ << ": weight updating ... ";
    // vec_t diff;
    // prev()->merge_grads(&diff);
#ifdef CPU_ONLY
    // parallelize only when target size is big enough to mitigate thread
    // spawning overhead.
    bool parallel = (W.size() >= 512);
    opt->update(weight_grad, W, parallel); // W += grad
#else
    opt->update_gpu(d_weight_grad, d_W); // W += grad
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
