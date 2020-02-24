#pragma once
#include <vector>
#include "types.h"
class node;
class layer;
class edge;

typedef std::shared_ptr<edge> edgeptr_t;

// node data structure
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
		data_(vec_t(n*len)), grad_(vec_t(n*len)),
		prev_(prev) {}

	void merge_grads(vec_t *dst) {
		assert(!grad_.empty());
		dst->resize(ft_dim_);
		float_t *pdst = &(*dst)[0];
		std::copy(grad_.begin(), grad_.begin()+ft_dim_, pdst);
		// @todo consider adding parallelism and vectorization
		for (size_t sample = 1; sample < num_samples_; ++sample) {
			for (size_t i = 0; i < ft_dim_; i++) pdst[i] += grad_[sample*ft_dim_+i];
			//vectorize::reduce<float_t>(&grad_[sample][0], ft_dim_, pdst);
		}
	}
	void clear_grads() {
		std::fill(grad_.begin(), grad_.end(), float_t{0}); // TODO: need vectorize
		//vectorize::fill(&grad_[0], grad_.size(), float_t{0});
	}

	vec_t &get_data() { return data_; }
	const vec_t &get_data() const { return data_; }
	vec_t &get_gradient() { return grad_; }
	const vec_t &get_gradient() const { return grad_; }
	float_t *get_gpu_data() const { return gpu_data_; }
	float_t *get_gpu_gradient() { return gpu_grad_; }

	const node *next() const { return next_; }
	node *prev() { return prev_; }
	const node *prev() const { return prev_; }
	void add_next_node(node *next) { next_ = next; }

private:
	size_t num_samples_;// number of samples
	size_t ft_dim_;     // feature dimensions
	vec_t data_;        // feature vectors on CPU
	vec_t grad_;        // gradients on CPU
	float_t *gpu_data_; // feature vectors on GPU
	float_t *gpu_grad_; // gradients on CPU
	node *prev_;        // previous node, "producer" of data
	node *next_;        // next node, "consumer" of data
};

