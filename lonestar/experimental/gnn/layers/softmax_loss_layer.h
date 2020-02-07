#pragma once
#include "layer.h"

class softmax_loss_layer: public layer {
public:
	softmax_loss_layer(unsigned level, std::vector<size_t> in_dims, 
		std::vector<size_t> out_dims, LabelList *lab)
		: layer(level, in_dims, out_dims), labels(lab) {
		trainable_ = false;
		loss.resize(in_dims[0]); // error for each sample
		name_ = layer_type() + "_" + std::to_string(level);
	}
	softmax_loss_layer(unsigned level, std::vector<size_t> in_dims, 
		std::vector<size_t> out_dims) : 
		softmax_loss_layer(level, in_dims, out_dims, NULL) {}
	~softmax_loss_layer() {}
	std::string layer_type() const override { return std::string("softmax_loss"); }

	// TODO: need kernel fusion optimization
	// 𝑦[i] = 𝑒^𝑥[i] / Σ 𝑒^𝑥[𝑘]
	void forward_propagation(const tensor_t &in_data, tensor_t &out_data) override {
		//std::cout << name_ << " forward: in_x=" << in_data.size() << ", in_y=" 
		//	<< in_data[0].size() << ", out_y=" << out_data[0].size() << "\n";
		galois::do_all(galois::iterate((size_t)0, output_dims[0]), [&](const auto& i) {
			if ((*labels)[i] >= 0) { // masked
				softmax(in_data[i], out_data[i]); // normalize using softmax
				// y is a one hot encoded vector for the labels
				std::vector<acc_t> y(output_dims[1], 0.0); // ground truth
				y[(*labels)[i]] = 1.0; // one-hot
				loss[i] = cross_entropy(y, out_data[i]);
				/*
				if (i < 10) {
					std::cout << "vertex " << i << " label: " << (*labels)[i] << ", pred: [ ";
					for (size_t j = 0; j < output_dims[1]; j++)
						std::cout << in_data[i][j] << " ";
					std::cout << "], normailized_pred: [ ";
					for (size_t j = 0; j < output_dims[1]; j++)
						std::cout << out_data[i][j] << " ";
					std::cout << "], loss: " << loss[i] << "\n";
				}
				//*/
			}
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("softmax_loss-fw"));
	}

	void back_propagation(const tensor_t &in_data, const tensor_t &out_data, tensor_t &out_grad, tensor_t &in_grad) override {
		//std::cout << name_ << " backward: x=" << in_grad.size() << ", y=" << in_grad[0].size() << "\n";
		galois::do_all(galois::iterate((size_t)0, output_dims[0]), [&](const auto& i) {
			vec_t norm_grad(output_dims[1]);
			std::vector<acc_t> y(output_dims[1], 0.0); // ground truth
			y[(*labels)[i]] = 1.0;
			d_cross_entropy(y, out_data[i], norm_grad);
			d_softmax(in_data[i], out_data[i], in_grad[i], norm_grad);
		}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("cross-entropy-bw"));
	}

private:
	LabelList *labels;
};

