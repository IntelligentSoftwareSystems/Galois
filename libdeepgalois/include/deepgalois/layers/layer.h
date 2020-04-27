#pragma once
/**
 * Code from on below link. Modified under Galois's license.
 *
 * https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/layer.h
 *
 * Copyright (c) 2013, Taiga Nomi and the respective contributors
 * All rights reserved.
 * Reused/revised under 3-BSD
 */

#include "deepgalois/types.h"
#ifndef GALOIS_USE_DIST
#include "deepgalois/context.h"
#else
#include "deepgalois/DistContext.h"
#endif
#include "deepgalois/optimizer.h"
#include "deepgalois/math_functions.hh"
#include "deepgalois/layers/node.h"
#ifdef GALOIS_USE_DIST
#include "galois/graphs/GluonSubstrate.h"
#include "deepgalois/layers/GluonGradients.h"
#include "deepgalois/layers/GradientSyncStructs.h"
#endif

namespace deepgalois {

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
class layer : public deepgalois::node {
public:
#ifndef GALOIS_USE_DIST
  using ContextType = deepgalois::Context;
#else
  using ContextType = deepgalois::DistContext;
#endif

  layer(unsigned level, std::vector<size_t> in_dims,
        std::vector<size_t> out_dims)
      : node(in_dims.size(), out_dims.size()), level_(level), begin_(0),
        end_(0), num_dims(in_dims.size()), input_dims(in_dims),
        output_dims(out_dims), labels(NULL) { }
  virtual ~layer()                       = default;
  virtual std::string layer_type() const = 0;
  void print_layer_info(); //! debug print function
  virtual void malloc_and_init() {}

  // get methods
  virtual acc_t get_prediction_loss() { return acc_t(0); }
  virtual acc_t get_weight_decay_loss() { return acc_t(0); }
  bool trainable() const { return trainable_; }
  std::string get_name() { return name_; }
  mask_t* get_device_masks() { return d_masks_; }
  float_t* get_weights_ptr() { return &W[0]; }
  float_t* get_weights_device_ptr() { return d_W; }
  float_t* get_grads_ptr() { return &weight_grad[0]; }
  float_t* get_grads_device_ptr() { return d_weight_grad; }

  // set methods
  virtual void set_netphase(net_phase phase) {}
  virtual void set_context(ContextType* ctx) { context = ctx; }
  void set_trainable(bool trainable) { trainable_ = trainable; } // is this layer trainable?
  void set_labels_ptr(label_t *ptr) { labels = ptr; }
  void set_name(std::string name) { name_ = name; } // name metadata
#ifdef CPU_ONLY
  void set_graph_ptr(Graph *ptr) { graph_cpu = ptr; }
#else
  void set_graph_ptr(CSRGraph *ptr) { graph_gpu = ptr; }
#endif
  void update_dim_size(size_t sg_size) { input_dims[0] = output_dims[0] = sg_size; }

  //! set the data of the previous layer connected to this one
  void set_in_data(float_t* data) {
    prev_ = std::make_shared<deepgalois::edge>(this, input_dims[0], input_dims[1]);
    prev_->set_data(data);
    // no need to allocate memory for gradients, since this is the input layer.
  }

  virtual void set_sample_mask(size_t sample_begin, size_t sample_end,
                               size_t sample_count, mask_t* masks) {
    begin_ = sample_begin;
    end_   = sample_end;
    count_ = sample_count;
#ifdef CPU_ONLY
    masks_ = masks;
#else
	d_masks_ = masks;
#endif
  }

  void add_edge() {
    // add an outgoing edge
    next_ = std::make_shared<deepgalois::edge>(this, output_dims[0], output_dims[1]);
    // allocate memory for intermediate feature vectors and gradients
    next_->alloc();
  }

  // main functions for layer work
  virtual void forward_propagation(const float_t* in_data,
                                   float_t* out_data)                = 0;
  virtual void back_propagation(const float_t* in_data, const float_t* out_data,
                                float_t* out_grad, float_t* in_grad) = 0;

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

  //! use optimizer to update weights given gradient (weight_grad)
  void update_weight(deepgalois::optimizer* opt) {
#ifdef CPU_ONLY
    // parallelize only when target size is big enough to mitigate thread
    // spawning overhead.
    bool parallel = (W.size() >= 512);
    opt->update(layer::weight_grad, layer::W, parallel); // W += grad
#else
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
  ContextType* context;
  label_t* labels;
#ifdef CPU_ONLY
  Graph *graph_cpu;
#else
  CSRGraph *graph_gpu;
#endif

#ifdef GALOIS_USE_DIST
  // Used for synchronization of weight gradients
  deepgalois::GluonGradients* gradientGraph;
  galois::graphs::GluonSubstrate<deepgalois::GluonGradients>* syncSub;
#endif
};


//! Connects tail to head's edge and sets that edge's target to tail
//inline void connect(layer* head, layer* tail) {
inline void connect(layer* head, layer* tail) {
  tail->prev_ = head->next_;
  tail->prev_->add_next_node(tail);
}

} // namespace deepgalois
