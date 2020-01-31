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
	graph_conv_layer(unsigned level, std::vector<size_t> in_dims, std::vector<size_t> out_dims)
		: layer(level, in_dims, out_dims) {
		trainable_ = true;
		// randomly initialize trainable parameters for conv layers
		init_matrix(input_dims[1], output_dims[1], W);
		init_matrix(input_dims[1], output_dims[1], Q);
	}
	~graph_conv_layer() {}
	std::string layer_type() const override { return std::string("graph_conv"); }
	void setup(Graph *g, FV *d, LabelList *lab) override { graph = g; }

	void forward(const std::vector<FV> &in_data, std::vector<FV> &out_data) override {
		size_t x = output_dims[0];
		size_t z = output_dims[1];
		FV2D fv_temp(x); // x * z
		for (size_t i = 0; i < x; ++i) fv_temp[i].resize(z);
		matmul(in_data, W, fv_temp); // x*y; y*z; x*z
		update_all(graph, z, fv_temp, out_data);
		if (act_) {
			galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
				relu(out_data[i], out_data[i]);
			}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("relu"));
		}
	}

	void backward(const std::vector<FV> &in_data, const std::vector<FV> &out_data, std::vector<FV> &out_grad, std::vector<FV> &in_grad) override {
		size_t x = in_grad.size();
		size_t z = out_data[0].size();
		assert(y == in_data[0].size());
		if (act_) {
			galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
				for (size_t j = 0; j < z; ++j) 
				d_relu(in_grad[i], out_data[i], out_grad[i]); // x*z
			}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("relu_back"));
		}
		//FV conv1_diff(num_classes, 0); // used to gather neighbors' gradients
		matmul(in_grad, in_data, out_grad); // x*z; z*y; x*y
		//d_update_all(out_grad, hidden1_diff[src]); // 16 x E; E x 1; hidden1_diff: N x 16 
	}

	void update_weights(optimizer *opt) override {
		bool parallel = true;
		//opt->update(grad, W, parallel); // W += diff
	}

private:
	Graph *graph;
	FV2D W; // parameters to learn, for vertex v, layer0: D x 16, layer1: 16 x E
	FV2D Q; // parameters to learn, for vertex u, i.e. v's neighbors, layer0: D x 16, layer1: 16 x E

	inline void init_matrix(size_t dim_x, size_t dim_y, FV2D &matrix) {
		// Glorot & Bengio (AISTATS 2010) init
		auto init_range = sqrt(6.0/(dim_x + dim_y));
		//std::cout << "Matrix init_range: (" << -init_range << ", " << init_range << ")\n";
		std::default_random_engine rng;
		std::uniform_real_distribution<FeatureT> dist(-init_range, init_range);
		matrix.resize(dim_x);
		for (size_t i = 0; i < dim_x; ++i) {
			matrix[i].resize(dim_y);
			for (size_t j = 0; j < dim_y; ++j)
				matrix[i][j] = dist(rng);
		}
		//for (size_t i = 0; i < 3; ++i)
		//	for (size_t j = 0; j < 3; ++j)
		//		std::cout << "matrix[" << i << "][" << j << "]: " << matrix[i][j] << std::endl;
	}
};
