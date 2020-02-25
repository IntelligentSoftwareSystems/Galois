#pragma once
#include <vector>
#include <memory>
#include <cassert>
#include "types.h"
class node;
class layer;
class edge;

typedef std::shared_ptr<edge> edgeptr_t;

// node data structure: each layer is a node, two layers are connected by an edge
class node : public std::enable_shared_from_this<node> {
public:
	node(size_t in_size, size_t out_size) {}//: prev_(in_size), next_(out_size) {}
	virtual ~node() {}
	const edgeptr_t prev() const { return prev_; }
	const edgeptr_t next() const { return next_; }

protected:
	node() = delete;
	friend void connect(layer *head, layer *tail, size_t head_index, size_t tail_index);
	mutable edgeptr_t prev_;
	mutable edgeptr_t next_;
};

// edges manage the input/output data and gradients between nodes
class edge {
public:
	edge(node *prev, size_t n, size_t len) :
		num_samples_(n), ft_dim_(len),
		data_(NULL), grad_(NULL), prev_(prev) {}

	void alloc();
	void alloc_gpu();
	void merge_grads(vec_t *dst);
	void merge_grads_gpu(float_t *dst);
	void clear_grads();
	void clear_grads_gpu();

	void set_data(float_t *ptr) { data_ = ptr; }
	float_t *get_data() { return data_; }
	const float_t *get_data() const { return data_; }
	float_t *get_gradient() { return grad_; }
	const float_t *get_gradient() const { return grad_; }

	const node *next() const { return next_; }
	node *prev() { return prev_; }
	const node *prev() const { return prev_; }
	void add_next_node(node *next) { next_ = next; }

private:
	size_t num_samples_;// number of samples
	size_t ft_dim_;     // feature dimensions
	float_t *data_; // feature vectors
	float_t *grad_; // gradients
	node *prev_;        // previous node, "producer" of data
	node *next_;        // next node, "consumer" of data
};

