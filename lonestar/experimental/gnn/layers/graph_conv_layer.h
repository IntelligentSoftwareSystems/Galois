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
	graph_conv_layer(unsigned level, Graph *g, bool dropout,
		std::vector<size_t> in_dims, std::vector<size_t> out_dims) :
		layer(level, in_dims, out_dims), graph(g) {
		assert(input_dims[0] == output_dims[0]); // num_vertices
		trainable_ = true;
		// randomly initialize trainable parameters for conv layers
		rand_init_matrix(input_dims[1], output_dims[1], W);
		//rand_init_matrix(input_dims[1], output_dims[1], Q);
		zero_init_matrix(input_dims[1], output_dims[1], weight_grad);
		name_ = layer_type() + "_" + std::to_string(level);
		alloc_grad();
		dropout_ = dropout;
		if (dropout_) {
			dropout_mask.resize(input_dims[0]);
			dropout_data.resize(input_dims[0]);
			for (size_t i = 0; i < input_dims[0]; i++) {
				dropout_mask[i].resize(input_dims[1]);
				dropout_data[i].resize(input_dims[1]);
			}
		}
		pre_sup.resize(output_dims[0]); // pre_sup: same as it is in original GCN implementation
		for (size_t i = 0; i < input_dims[0]; ++i) pre_sup[i].resize(output_dims[1]);
	}
	graph_conv_layer(unsigned level, std::vector<size_t> in_dims, 
		std::vector<size_t> out_dims) : graph_conv_layer(level, NULL, false, in_dims, out_dims) {}

	~graph_conv_layer() {}
	std::string layer_type() const override { return std::string("graph_conv"); }
	//void setup(Graph *g, vec_t *d, LabelList *lab) override { graph = g; }

	// 𝒉[𝑙] = σ(𝑊 * Σ(𝒉[𝑙-1]))
	void forward_propagation(const tensor_t &in_data, tensor_t &out_data) override {
		// if in_feats_dim > out_feats_dim:
		// mult W first to reduce the feature size for aggregation
		// else: aggregate first then mult W (not implemented yet)
		size_t x = output_dims[0]; // input: x*y; W: y*z; output: x*z
		if (dropout_) {
			for (size_t i = 0; i < x; ++i) {
			//galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
				dropout(in_data[i], dropout_mask[i], dropout_data[i]);
			}//, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("dropout"));
			matmul(dropout_data, W, pre_sup);
		} else matmul(in_data, W, pre_sup);
		update_all(graph, pre_sup, out_data); // aggregate
		if (act_) {
			galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
				relu(out_data[i], out_data[i]);
			}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("relu"));
		}
	}

	// 𝜕𝐸 / 𝜕𝑦[𝑙−1] = 𝜕𝐸 / 𝜕𝑦[𝑙] ∗ 𝑊 ^𝑇
	void back_propagation(const tensor_t &in_data, const tensor_t &out_data, tensor_t &out_grad, tensor_t &in_grad) override {
		size_t x = output_dims[0];
		size_t y = input_dims[1];
		size_t z = output_dims[1];
		tensor_t grad_temp = out_grad;
		if (act_) {
			//for (size_t j = 0; j < z; ++j) 
			galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
				for (size_t j = 0; j < z; ++j) 
					//grad_temp[i*z+j] = out_grad[i][j] * (out_data[i][j] > 0.0);
					if (out_data[i][j] <= 0.0) grad_temp[i][j] = 0.0;
			}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("d_relu"));
		}
		vec_t trans_W(z*y);
		transpose(y, z, W, trans_W);
		matmul(grad_temp, trans_W, in_grad); // x*z; z*y; x*y
		//d_update_all(out_grad, ); //
		if (dropout_) {
			galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
				d_dropout(in_grad[i], dropout_mask[i], in_grad[i]);
			}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("d_dropout"));
		}

		// calculate weight gradients
		tensor_t trans_data(y); // y*x
		for (size_t i = 0; i < y; ++i) trans_data[i].resize(x);
		transpose2D(in_data, trans_data);
		matmul2D1D(trans_data, grad_temp, weight_grad); // y*x; x*z; y*z
		/*
		for (size_t i = 0; i < 3; i ++)
			for (size_t j = 0; j < 2; j ++) {
				std::cout << "out_data[" << i << "][" << j << "]=" << out_data[i][j] << "\n";
				std::cout << "out_grad[" << i << "][" << j << "]=" << out_grad[i][j] << "\n";
				std::cout << "grad_temp[" << i << "][" << j << "]=" << grad_temp[i][j] << "\n";
				std::cout << "trans_data[" << i << "][" << j << "]=" << trans_data[i][j] << "\n";
			}
		for (size_t i = 0; i < 6; i ++) std::cout << "weight_grad[" << i << "]=" << weight_grad[i] << "\n";
		//*/
	}

private:
	Graph *graph;
	tensor_t pre_sup;
	tensor_t dropout_data;
	std::vector<std::vector<unsigned> > dropout_mask;

	// Glorot & Bengio (AISTATS 2010) init
	inline void rand_init_matrix(size_t dim_x, size_t dim_y, vec_t &matrix) {
		auto init_range = sqrt(6.0/(dim_x + dim_y));
		//std::cout << "Matrix init_range: (" << -init_range << ", " << init_range << ")\n";
		std::default_random_engine rng;
		std::uniform_real_distribution<float_t> dist(-init_range, init_range);
		matrix.resize(dim_x * dim_y);
		for (size_t i = 0; i < dim_x; ++i) {
			for (size_t j = 0; j < dim_y; ++j)
				matrix[i*dim_y+j] = dist(rng);
		}
	}
	inline void zero_init_matrix(size_t dim_x, size_t dim_y, vec_t &matrix) {
		matrix.resize(dim_x * dim_y);
		for (size_t i = 0; i < dim_x; ++i) {
			for (size_t j = 0; j < dim_y; ++j)
				matrix[i*dim_y+j] = 0;
		}
	}
};
