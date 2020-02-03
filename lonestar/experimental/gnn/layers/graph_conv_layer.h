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
	graph_conv_layer(unsigned level, Graph *g,
		std::vector<size_t> in_dims, std::vector<size_t> out_dims) :
		layer(level, in_dims, out_dims), graph(g) {
		trainable_ = true;
		// randomly initialize trainable parameters for conv layers
		init_matrix(input_dims[1], output_dims[1], W);
		init_matrix(input_dims[1], output_dims[1], Q);
		name_ = layer_type() + "_" + std::to_string(level);
		alloc_grad();
	}
	graph_conv_layer(unsigned level, std::vector<size_t> in_dims, 
		std::vector<size_t> out_dims) : graph_conv_layer(level, NULL, in_dims, out_dims) {}

	~graph_conv_layer() {}
	std::string layer_type() const override { return std::string("graph_conv"); }
	//void setup(Graph *g, vec_t *d, LabelList *lab) override { graph = g; }

	// 𝒉[𝑙] = σ(𝑊 * Σ(𝒉[𝑙-1]))
	void forward_propagation(const tensor_t &in_data, tensor_t &out_data) override {
		//std::cout << name_ << " forward: in_x=" << in_data.size() << ", in_y=" 
		//	<< in_data[0].size() << ", out_y=" << out_data[0].size() << "\n";
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

	// 𝜕𝐸 / 𝜕𝑦[𝑙−1] = 𝜕𝐸 / 𝜕𝑦[𝑙] ∗ 𝑊 ^𝑇
	void back_propagation(const tensor_t &in_data, const tensor_t &out_data, tensor_t &out_grad, tensor_t &in_grad) override {
		//std::cout << name_ << " backward: x=" << in_grad.size() << ", y=" << in_grad[0].size() << "\n";
		size_t x = output_dims[0];
		size_t y = input_dims[1];
		size_t z = output_dims[1];
		//vec_t grad_temp(x*z);
		tensor_t grad_temp(x);
		for (size_t i = 0; i < x; ++i) grad_temp[i].resize(z);
		if (act_) {
			//for (size_t j = 0; j < z; ++j) 
			galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
				for (size_t j = 0; j < z; ++j) 
					//grad_temp[i*z+j] = out_grad[i][j] * (out_data[i][j] > 0.0);
					grad_temp[i][j] = out_grad[i][j] * (out_data[i][j] > 0.0);
			}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("d_relu"));
		}
		//tensor_t trans_data(y); // y*x
		//for (size_t i = 0; i < y; ++i) trans_data[i].resize(x);
		//transpose(in_data, trans_data);
		//matmul(trans_data, grad_temp, in_grad); // y*x; x*z; y*z
		vec_t trans_W(z*y);
		transpose(y, z, W, trans_W);
		matmul(grad_temp, trans_W, in_grad); // x*z; z*y; x*y
		//d_update_all(out_grad, ); //
	}

private:
	Graph *graph;

	// Glorot & Bengio (AISTATS 2010) init
	inline void init_matrix(size_t dim_x, size_t dim_y, vec_t &matrix) {
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
