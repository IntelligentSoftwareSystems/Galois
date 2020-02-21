#include "layers/softmax_loss_layer.h"

softmax_loss_layer::softmax_loss_layer(unsigned level, std::vector<size_t> in_dims, std::vector<size_t> out_dims, LabelList *lab)
	: layer(level, in_dims, out_dims), labels(lab) {
	trainable_ = false;
	loss.resize(in_dims[0]); // error for each sample
	name_ = layer_type() + "_" + std::to_string(level);
}

// TODO: need kernel fusion optimization
// 𝑦[i] = 𝑒^𝑥[i] / Σ 𝑒^𝑥[𝑘]
void softmax_loss_layer::forward_propagation(const tensor_t &in_data, tensor_t &out_data) {
	galois::do_all(galois::iterate(begin_, end_), [&](const auto& i) {
		if (masks_[i] == 1) { // masked
			softmax(in_data[i], out_data[i]); // normalize using softmax
			// y is a one hot encoded vector for the labels
			std::vector<acc_t> y(output_dims[1], 0.0); // ground truth
			y[(*labels)[i]] = 1.0; // one-hot
			loss[i] = cross_entropy(y, out_data[i]);
		}
	}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("softmax-loss-fw"));
}

void softmax_loss_layer::back_propagation(const tensor_t &in_data, const tensor_t &out_data, tensor_t &out_grad, tensor_t &in_grad) {
	//std::cout << name_ << " backward: x=" << in_grad.size() << ", y=" << in_grad[0].size() << "\n";
	galois::do_all(galois::iterate(begin_, end_), [&](const auto& i) {
		vec_t norm_grad(output_dims[1]);
		std::vector<acc_t> y(output_dims[1], 0.0); // ground truth
		y[(*labels)[i]] = 1.0;
		d_cross_entropy(y, out_data[i], norm_grad);
		d_softmax(in_data[i], out_data[i], in_grad[i], norm_grad);
	}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("softmax-loss-bw"));
}

