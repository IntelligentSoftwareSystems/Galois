#pragma once
#include "layer.h"

class softmax_loss_layer: public layer {
public:
	softmax_loss_layer(unsigned level, std::vector<size_t> in_dims, 
		std::vector<size_t> out_dims, vec_t *d, LabelList *lab)
		: layer(level, in_dims, out_dims), diffs(d), labels(lab) {
		trainable_ = false;
		name_ = layer_type() + "_" + std::to_string(level);
	}
	softmax_loss_layer(unsigned level, std::vector<size_t> in_dims, 
		std::vector<size_t> out_dims) : 
		softmax_loss_layer(level, in_dims, out_dims, NULL, NULL) {}
	~softmax_loss_layer() {}
	std::string layer_type() const override { return std::string("softmax_loss"); }
	//void setup(Graph *g, vec_t *d, LabelList *lab) override { diffs = d; labels = lab; }

	// TODO: need kernel fusion optimization
	void forward_propagation(const tensor_t &in_data, tensor_t &out_data) override {
		//std::cout << name_ << " forward: in_x=" << in_data.size() << ", in_y=" 
		//	<< in_data[0].size() << ", out_y=" << out_data[0].size() << "\n";
		galois::do_all(galois::iterate((size_t)0, output_dims[0]), [&](const auto& i) {
			if ((*labels)[i] >= 0) { // masked
				softmax(in_data[i], out_data[i]); // normalize using softmax
				// y is a one hot encoded vector for the labels
				std::vector<acc_t> y(output_dims[1], 0.0); // ground truth
				y[(*labels)[i]] = 1.0; // one-hot
				(*diffs)[i] = cross_entropy(y, out_data[i]);
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("softmax_loss-fw"));
	}

	void back_propagation(const tensor_t &in_data, const tensor_t &out_data, tensor_t &out_grad, tensor_t &in_grad) override {
		//std::cout << name_ << " backward: x=" << in_grad.size() << ", y=" << in_grad[0].size() << "\n";
		galois::do_all(galois::iterate((size_t)0, output_dims[0]), [&](const auto& i) {
			in_grad[i].resize(output_dims[1]);
			std::vector<acc_t> y(output_dims[1], 0.0); // ground truth
			y[(*labels)[i]] = 1.0;
			d_cross_entropy(y, out_data[i], in_grad[i]);
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("cross-entropy-bw"));
	}

private:
	vec_t *diffs;
	LabelList *labels;
};

