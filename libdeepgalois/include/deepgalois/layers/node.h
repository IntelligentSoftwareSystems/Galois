#pragma once
/**
 * Code modified from below
 *
 * https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/node.h
 *
 * Copyright (c) 2013, Taiga Nomi and the respective contributors
 * All rights reserved.
 * Reused/revised under 3-BSD
 */

#include <vector>
#include <memory>
#include <cassert>
#include "deepgalois/types.h"

namespace deepgalois {

class node;
class layer;
class edge;

typedef std::shared_ptr<edge> edgeptr_t;

// node data structure: each layer is a node, two layers are connected by an
// edge
class node : public std::enable_shared_from_this<node> {
public:
  node() {
    prev_ = NULL;
    next_ = NULL;
  }
  // node(size_t in_size, size_t out_size) {
  //} //: prev_(in_size), next_(out_size) {}
  virtual ~node() {}
  const edgeptr_t prev() const { return prev_; }
  const edgeptr_t next() const { return next_; }

protected:
  // node() = delete;
  friend void connect(layer* head, layer* tail);
  mutable edgeptr_t prev_;
  mutable edgeptr_t next_;
};

// edges manage the input/output data and gradients between nodes
class edge {
public:
  edge(node* prev, size_t n, size_t len)
      : num_samples_(n), ft_dim_(len), data_(NULL), grad_(NULL), prev_(prev) {}

  void alloc();
  void clear_grads();
  void merge_grads(float_t* dst);
  void set_data(float_t* ptr) { data_ = ptr; }
  float_t* get_data() { return data_; }
  const float_t* get_data() const { return data_; }
  float_t* get_gradient() { return grad_; }
  const float_t* get_gradient() const { return grad_; }

  const node* next() const { return next_; }
  node* prev() { return prev_; }
  const node* prev() const { return prev_; }
  void add_next_node(node* next) { next_ = next; }

private:
  size_t num_samples_; // number of samples
  size_t ft_dim_;      // feature dimensions
  float_t* data_;      // feature vectors
  float_t* grad_;      // gradients
  node* prev_;         // previous node, "producer" of data
  node* next_;         // next node, "consumer" of data
};

} // namespace deepgalois
