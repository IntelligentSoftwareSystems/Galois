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
	void setup(Graph *g, vec_t *d, LabelList *lab) override { graph = g; }

	void forward_propagation(const tensor_t &in_data, tensor_t &out_data) override {
		size_t x = output_dims[0];
		size_t z = output_dims[1];
		tensor_t fv_temp(x); // x * z
		for (size_t i = 0; i < x; ++i) fv_temp[i].resize(z);
		matmul(in_data, W, fv_temp); // x*y; y*z; x*z
		update_all(graph, z, fv_temp, out_data);
		if (act_) {
			galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
				relu(out_data[i], out_data[i]);
			}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("relu"));
		}
	}

	void back_propagation(const tensor_t &in_data, const tensor_t &out_data, tensor_t &out_grad, tensor_t &in_grad) override {
		size_t x = output_dims[0];
		size_t y = input_dims[1];
		size_t z = output_dims[1];
		vec_t grad_temp(x*z);
		if (act_) {
			//for (size_t j = 0; j < z; ++j) 
			galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
				for (size_t j = 0; j < z; ++j) 
					grad_temp[i*z+j] = in_grad[i][j] * (out_data[i][j] > 0.0);
			}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("d_relu"));
		}
		tensor_t trans_in_data(y); // y*x
		for (size_t i = 0; i < y; ++i) trans_in_data[i].resize(x);
		transpose(in_data, trans_in_data);
		matmul(trans_in_data, grad_temp, out_grad); // y*x; x*z; y*z
		//d_update_all(out_grad, hidden1_diff[src]); // 16 x E; E x 1; hidden1_diff: N x 16 
	}

	void update_weights(optimizer *opt) override {
		// parallelize only when target size is big enough to mitigate thread spawning overhead.
		bool parallel = (W.size() >= 512);
		vec_t diff;
		prev()->merge_grads(&diff);
		auto in_data = prev()->get_data();
		float_t rcp_batch_size = float_t(1.0) / in_data.size();
		for (size_t i = 0; i < diff.size(); ++i)
			diff[i] *= rcp_batch_size;
		opt->update(diff, W, parallel); // W += diff
		prev()->clear_grads();
	}

private:
	Graph *graph;
	vec_t W; // parameters to learn, for vertex v, layer0: D x 16, layer1: 16 x E
	vec_t Q; // parameters to learn, for vertex u, i.e. v's neighbors, layer0: D x 16, layer1: 16 x E

	inline void init_matrix(size_t dim_x, size_t dim_y, vec_t &matrix) {
		// Glorot & Bengio (AISTATS 2010) init
		auto init_range = sqrt(6.0/(dim_x + dim_y));
		//std::cout << "Matrix init_range: (" << -init_range << ", " << init_range << ")\n";
		std::default_random_engine rng;
		std::uniform_real_distribution<float_t> dist(-init_range, init_range);
		matrix.resize(dim_x * dim_y);
		for (size_t i = 0; i < dim_x; ++i) {
			for (size_t j = 0; j < dim_y; ++j)
				matrix[i*dim_y+j] = dist(rng);
		}
		//for (size_t i = 0; i < 3; ++i)
		//	for (size_t j = 0; j < 3; ++j)
		//		std::cout << "matrix[" << i << "][" << j << "]: " << matrix[i][j] << std::endl;
	}
};
