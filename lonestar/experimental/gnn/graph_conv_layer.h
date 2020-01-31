#pragma once
#include "layer.h"

/* GraphConv Layer
	Parameters
	----------
	in_feats : int, Input feature size.
	out_feats : int, Output feature size.
	norm : bool, optional, if True, the normalizer :math:`c_{ij}` is applied. Default: ``True``.
	bias : bool, optional, if True, adds a learnable bias to the output. Default: ``True``.
	activation: callable activation function/layer or None, optional
	If not None, applies an activation function to the updated node features. Default: ``None``.
*/
class graph_conv_layer: public layer {
public:
	graph_conv_layer() { act = true; }
	graph_conv_layer(bool act_) { act = act_; }
	~graph_conv_layer() {}
	std::string layer_type() const override { return std::string("graph_conv"); }

	void forward(const std::vector<FV> &in_data, std::vector<FV> &out_data) override {
		size_t x = in_data.size();
		size_t y = in_data[0].size();
		size_t z = (*W)[0].size();
		assert(y == (*W).size());
		FV2D fv_temp(x); // x * z
		for (size_t i = 0; i < x; ++i) fv_temp[i].resize(z);
		matmul(in_data, *W, fv_temp); // x*y; y*z; x*z
		update_all(graph, z, fv_temp, out_data);
		if (act) {
			galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
				relu(out_data[i], out_data[i]);
			}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("relu0"));
		}
	}

	void backward(const std::vector<FV> &in_data, const std::vector<FV> &out_data, std::vector<FV> &out_grad, std::vector<FV> &in_grad) override {
		size_t x = in_grad.size();
		size_t y = out_grad.size();
		size_t z = out_data[0].size();
		assert(y == in_data[0].size());
		if (act) {
			galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
				for (size_t j = 0; j < z; ++j) 
				d_relu(in_grad[i], out_data[i], out_grad[i]); // x*z
			}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("relu_back"));
		}
		//FV conv1_diff(num_classes, 0); // used to gather neighbors' gradients
		matmul(in_grad, in_data, out_grad); // x*z; z*y; x*y
		//d_update_all(out_grad, hidden1_diff[src]); // 16 x E; E x 1; hidden1_diff: N x 16 
	}

	void set_param(Graph *g, FV2D *w, FV2D *q, FV *d, LabelList *lab) {
		graph = g;
		W = w;
		Q = q;
	}

	void set_act(bool need_act) {
		act = need_act;
	}

private:
	Graph *graph;
	FV2D *W; // parameters
	FV2D *Q; // parameters
	bool act;
};
