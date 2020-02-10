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
			in_temp.resize(input_dims[0]);
			for (size_t i = 0; i < input_dims[0]; i++) {
				dropout_mask[i].resize(input_dims[1]);
				in_temp[i].resize(input_dims[1]);
			}
		}
		out_temp.resize(output_dims[0]); // same as pre_sup in original GCN code: https://github.com/chenxuhao/gcn/blob/master/gcn/layers.py
		for (size_t i = 0; i < input_dims[0]; ++i) out_temp[i].resize(output_dims[1]);
	}
	graph_conv_layer(unsigned level, std::vector<size_t> in_dims, 
		std::vector<size_t> out_dims) : graph_conv_layer(level, NULL, false, in_dims, out_dims) {}
	~graph_conv_layer() {}
	std::string layer_type() const override { return std::string("graph_conv"); }

	// user-defined aggregate function
	void aggregate(Graph *g, const tensor_t &in, tensor_t &out) { update_all(g, in, out); }

	// user-defined combine function
	void combine(const vec_t &self, const vec_t &neighbors, const vec_t mat_v, const vec_t mat_u, vec_t &out) {
		vec_t a(out.size(), 0);
		vec_t b(out.size(), 0);
		mvmul(mat_v, self, a);
		mvmul(mat_u, neighbors, b); 
		vadd(a, b, out); // out = W*self + Q*neighbors
	}

	// 𝒉[𝑙] = σ(𝑊 * Σ(𝒉[𝑙-1]))
	void forward_propagation(const tensor_t &in_data, tensor_t &out_data) override {
		// if in_feats_dim > out_feats_dim:
		// mult W first to reduce the feature size for aggregation
		// else: aggregate first then mult W (not implemented yet)
		size_t x = output_dims[0]; // input: x*y; W: y*z; output: x*z
		if (dropout_) {
			for (size_t i = 0; i < x; ++i) {
			//galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
				dropout(in_data[i], dropout_mask[i], in_temp[i]);
			}//, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("dropout"));
			matmul(in_temp, W, out_temp); // x*y; y*z; x*z
		} else matmul(in_data, W, out_temp); // matrix multiply feature vector
		aggregate(graph, out_temp, out_data); // aggregate
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
		out_temp = out_grad;
		if (act_) {
			//for (size_t j = 0; j < z; ++j) 
			galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
				for (size_t j = 0; j < z; ++j) 
					if (out_data[i][j] <= 0.0) out_temp[i][j] = 0.0;
			}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("d_relu"));
		}
		if (level_ != 0) { // no need to calculate in_grad for the first layer
			vec_t trans_W(z*y);
			transpose(y, z, W, trans_W); // derivative of matmul needs transposed matrix
			matmul(out_temp, trans_W, in_temp); // x*z; z*y -> x*y
			update_all(graph, in_temp, in_grad); // x*x; x*y -> x*y NOTE: since graph is symmetric, the derivative is the same
			if (dropout_) {
				galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
					d_dropout(in_grad[i], dropout_mask[i], in_grad[i]);
				}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("d_dropout"));
			}
		}

		// calculate weight gradients
		tensor_t trans_data(y); // y*x
		for (size_t i = 0; i < y; ++i) trans_data[i].resize(x);
		transpose2D(in_data, trans_data);
		matmul2D1D(trans_data, out_temp, weight_grad); // y*x; x*z; y*z
	}

private:
	Graph *graph;
	tensor_t out_temp;
	tensor_t in_temp;
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
