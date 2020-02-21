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
	edge(node *prev, size_t len) :
		ft_dim_(len),
		data_({vec_t(len)}),
		grad_({vec_t(len)}),
		prev_(prev) {}

	void merge_grads(vec_t *dst) {
		assert(!grad_.empty());
		const auto &grad_head = grad_[0];
		size_t sz             = grad_head.size();
		dst->resize(sz);
		float_t *pdst = &(*dst)[0];
		std::copy(grad_head.begin(), grad_head.end(), pdst);
		// @todo consider adding parallelism and vectorization
		for (size_t sample = 1; sample < grad_.size(); ++sample) {
			for (size_t i = 0; i < sz; i++) pdst[i] += grad_[sample][i];
			//vectorize::reduce<float_t>(&grad_[sample][0], sz, pdst);
		}
	}
	void clear_grads() {
		for (size_t sample = 0; sample < grad_.size(); ++sample) {
			auto &g = grad_[sample];
			std::fill(g.begin(), g.end(), 0.0); // TODO: need vectorize
			//vectorize::fill(&g[0], g.size(), float_t{0});
		}
	}

	tensor_t *get_data_ptr() { return &data_; }
	tensor_t &get_data() { return data_; }
	const tensor_t &get_data() const { return data_; }
	tensor_t &get_gradient() { return grad_; }
	const tensor_t &get_gradient() const { return grad_; }
	float_t *get_gpu_data() const { return gpu_data_; }
	float_t *get_gpu_gradient() { return gpu_grad_; }

	const node *next() const { return next_; }
	node *prev() { return prev_; }
	const node *prev() const { return prev_; }
	void add_next_node(node *next) { next_ = next; }

private:
	size_t ft_dim_;     // feature dimensions
	tensor_t data_;     // feature vectors on CPU
	tensor_t grad_;     // gradients on CPU
	float_t *gpu_data_; // feature vectors on GPU
	float_t *gpu_grad_; // gradients on CPU
	node *prev_;        // previous node, "producer" of this tensor
	node *next_;        // next node, "consumer" of this tensor
};

