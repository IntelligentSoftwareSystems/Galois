#pragma once
#include "layer.h"

class softmax_loss_layer: public layer {
public:
	softmax_loss_layer() {}
	~softmax_loss_layer() {}
	std::string layer_type() const override { return std::string("softmax_loss"); }

	// TODO: need kernel fusion optimization
	void forward(const std::vector<FV> &in_data, std::vector<FV> &out_data) override {
		size_t n = in_data.size();
		size_t num_classes = in_data[0].size();
		galois::do_all(galois::iterate((size_t)0, n), [&](const auto& i) {
			if ((*labels)[i] >= 0) { // masked
				softmax(in_data[i], out_data[i]); // normalize using softmax
				// y is a one hot encoded vector for the labels
				std::vector<AccT> y(num_classes, 0.0); // ground truth
				y[(*labels)[i]] = 1.0; // one-hot
				(*diffs)[i] = cross_entropy(y, out_data[i]);
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("softmax_loss"));
	}

	void backward(const std::vector<FV> &in_data, const std::vector<FV> &out_data, std::vector<FV> &out_grad, std::vector<FV> &in_grad) override {
		size_t n = in_data.size();
		size_t num_classes = in_data[0].size();
		galois::do_all(galois::iterate((size_t)0, n), [&](const auto& i) {
			in_grad[i].resize(num_classes);
			std::vector<AccT> y(num_classes, 0.0); // ground truth
			y[(*labels)[i]] = 1.0;
			d_cross_entropy(y, out_data[i], in_grad[i]);
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("cross-entropy-back"));
	}

	void set_param(Graph *g, FV2D *w, FV2D *q, FV *d, LabelList *lab) override {
		diffs = d;
		labels = lab;
	}

	void update_weights(optimizer *opt) override {}

private:
	FV *diffs;
	LabelList *labels;
};

