#pragma once
#include "layer.h"

class softmax_loss_layer: public layer {
public:
	softmax_loss_layer(unsigned level, std::vector<size_t> in_dims, std::vector<size_t> out_dims)
		: layer(level, in_dims, out_dims) {
		trainable_ = false;
	}
	~softmax_loss_layer() {}
	std::string layer_type() const override { return std::string("softmax_loss"); }
	void setup(Graph *g, vec_t *d, LabelList *lab) override { diffs = d; labels = lab; }

	// TODO: need kernel fusion optimization
	void forward_propagation(const tensor_t &in_data, tensor_t &out_data) override {
		size_t n = in_data.size();
		size_t num_classes = in_data[0].size();
		galois::do_all(galois::iterate((size_t)0, n), [&](const auto& i) {
			if ((*labels)[i] >= 0) { // masked
				softmax(in_data[i], out_data[i]); // normalize using softmax
				// y is a one hot encoded vector for the labels
				std::vector<acc_t> y(num_classes, 0.0); // ground truth
				y[(*labels)[i]] = 1.0; // one-hot
				(*diffs)[i] = cross_entropy(y, out_data[i]);
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("softmax_loss"));
	}

	void back_propagation(const tensor_t &in_data, const tensor_t &out_data, tensor_t &out_grad, tensor_t &in_grad) override {
		size_t n = in_data.size();
		size_t num_classes = in_data[0].size();
		galois::do_all(galois::iterate((size_t)0, n), [&](const auto& i) {
			in_grad[i].resize(num_classes);
			std::vector<acc_t> y(num_classes, 0.0); // ground truth
			y[(*labels)[i]] = 1.0;
			d_cross_entropy(y, out_data[i], in_grad[i]);
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("cross-entropy-back"));
	}

	void update_weights(optimizer *opt) override {}

private:
	vec_t *diffs;
	LabelList *labels;
};

