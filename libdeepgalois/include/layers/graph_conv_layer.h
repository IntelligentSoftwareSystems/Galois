#pragma once
#include "layer.h"

/* GraphConv Layer
	Parameters
	----------
	x: int, number of samples.
	y: int, Input feature size.
	z: int, Output feature size.
	dropout: bool, optional, if True, a dropout operation is applied before other operations.
	norm : bool, optional, if True, the normalizer :math:`c_{ij}` is applied. Default: ``True``.
	bias : bool, optional, if True, adds a learnable bias to the output. Default: ``False``.
	activation: callable activation function/layer or None, optional
	If not None, applies an activation function to the updated node features. Default: ``None``.
*/
class graph_conv_layer: public layer {
public:
	graph_conv_layer(unsigned level, Graph *g, bool act, bool norm, bool bias, bool dropout,
		float dropout_rate, std::vector<size_t> in_dims, std::vector<size_t> out_dims) :
		layer(level, in_dims, out_dims), graph(g), act_(act), norm_(norm), bias_(bias), 
		dropout_(dropout), dropout_rate_(dropout_rate) {
		assert(input_dims[0] == output_dims[0]); // num_vertices
		x = input_dims[0];
		y = input_dims[1];
		z = output_dims[1];
		trainable_ = true;
		name_ = layer_type() + "_" + std::to_string(level);
		//std::cout << name_ << " constructed: act(" << act_ << ") dropout(" << dropout << ")\n";
		init();
		scale_ = 1. / (1. - dropout_rate_);
	}
	graph_conv_layer(unsigned level, std::vector<size_t> in_dims, 
		std::vector<size_t> out_dims) : graph_conv_layer(level, NULL, false, true, false, true, 0.5, in_dims, out_dims) {}
	~graph_conv_layer() {}
	void init() {
		std::cout << name_ << ": allocating memory for parameters and intermediate data... ";
		Timer t_alloc;
		t_alloc.Start();
		// randomly initialize trainable parameters for conv layers
		rand_init_matrix(y, z, W);
		//rand_init_matrix(y, z, Q);
		zero_init_matrix(y, z, weight_grad);
		alloc_grad();
		if (dropout_) {
			dropout_mask.resize(x);
			for (size_t i = 0; i < x; i++) dropout_mask[i].resize(y);
		}
		in_temp.resize(x*y);
		//for (size_t i = 0; i < x; ++i) in_temp[i].resize(y);
		out_temp.resize(x*z); // same as pre_sup in original GCN code: https://github.com/chenxuhao/gcn/blob/master/gcn/layers.py
		//for (size_t i = 0; i < x; ++i) out_temp[i].resize(z);
		trans_data.resize(y*x); // y*x
		//for (size_t i = 0; i < y; ++i) trans_data[i].resize(x);
		if (norm_) norm_factor_counting();
		t_alloc.Stop();
		std::cout << "Done, allocation time: " << t_alloc.Millisecs() << " ms\n";
	}
	std::string layer_type() const override { return std::string("graph_conv"); }

	// user-defined aggregate function
	void aggregate(Graph *g, const vec_t &in, tensor_t &out) { update_all(g, in, out, true, norm_factor); }

	// user-defined combine function
	void combine(const vec_t &self, const vec_t &neighbors, const vec_t mat_v, const vec_t mat_u, vec_t &out) {
		vec_t a(out.size(), 0);
		vec_t b(out.size(), 0);
		mvmul(mat_v, self, a);
		mvmul(mat_u, neighbors, b); 
		vadd(a, b, out); // out = W*self + Q*neighbors
	}

	void set_context(net_phase ctx) override { phase_ = ctx; }

	// 𝒉[𝑙] = σ(𝑊 * Σ(𝒉[𝑙-1]))
	void forward_propagation(const tensor_t &in_data, tensor_t &out_data) override {
		// input: x*y; W: y*z; output: x*z
		// if y > z:
		// mult W first to reduce the feature size for aggregation
		// else: aggregate first then mult W (not implemented yet)
		//Timer t_matmul, t_agg, t_dropout;
		//t_matmul.Start();
		if (dropout_ && phase_ == net_phase::train) {
			//for (size_t i = 0; i < x; ++i) {
			galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
				dropout(scale_, dropout_rate_, in_data[i], dropout_mask[i], &in_temp[i*y]);
			}, galois::loopname("dropout"));
			matmul1D1D(x, z, y, in_temp, W, out_temp); // x*y; y*z; x*z
		} else matmul2D1D(z, in_data, W, out_temp); // x*y; y*z; x*z
		//t_matmul.Stop();
		//t_agg.Start();
		aggregate(graph, out_temp, out_data); // aggregate
		//t_agg.Stop();
		if (act_) {
			galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
				relu(out_data[i], out_data[i]);
			}, galois::loopname("relu"));
		}
		//double dropout_time = 0;
		//if (dropout_ && phase_ == net_phase::train) dropout_time = t_dropout.Millisecs();
		//std::cout << "\n\t" << name_ << " matmul time: " << t_matmul.Millisecs() 
		//	<< ", aggregation time: " << t_agg.Millisecs() << ", dropout time: " << dropout_time << "\n";
	}

	// 𝜕𝐸 / 𝜕𝑦[𝑙−1] = 𝜕𝐸 / 𝜕𝑦[𝑙] ∗ 𝑊 ^𝑇
	void back_propagation(const tensor_t &in_data, const tensor_t &out_data, tensor_t &out_grad, tensor_t &in_grad) override {
		if (act_) {
			//for (size_t j = 0; j < z; ++j) 
			galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
				for (size_t j = 0; j < z; ++j) 
					//if (out_data[i][j] <= 0.0) out_temp[i][j] = 0.0;
					out_temp[i*z+j] = out_data[i][j] > float_t(0) ? out_grad[i][j] : float_t(0);
			}, galois::loopname("d_relu"));
		//} else out_temp = out_grad; // TODO: avoid copying
		} else copy2D1D(out_grad, out_temp);
		if (level_ != 0) { // no need to calculate in_grad for the first layer
			vec_t trans_W(z*y);
			transpose(y, z, W, trans_W); // derivative of matmul needs transposed matrix
			matmul1D1D(x, y, z, out_temp, trans_W, in_temp); // x*z; z*y -> x*y
			update_all(graph, in_temp, in_grad, true, norm_factor); // x*x; x*y -> x*y NOTE: since graph is symmetric, the derivative is the same
			if (dropout_) {
				galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
					d_dropout(scale_, in_grad[i], dropout_mask[i], in_grad[i]);
				}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("d_dropout"));
			}
		}

		// calculate weight gradients
		transpose2D1D(in_data, trans_data); // y*x
		matmul1D1D(y, z, x, trans_data, out_temp, weight_grad); // y*x; x*z; y*z
	}

	void degree_counting() {
		assert(x == graph->size());
		degrees.resize(x);
		galois::do_all(galois::iterate((size_t)0, x), [&] (auto v) {
			degrees[v] = std::distance(graph->edge_begin(v), graph->edge_end(v));
		}, galois::loopname("DegreeCounting"));
	}

	// for each vertex v, compute pow(|N(v)|, -0.5), where |N(v)| is the degree of v
	void norm_factor_counting() {
		degree_counting();
		norm_factor.resize(x);
		galois::do_all(galois::iterate((size_t)0, x), [&] (auto v) {
			float_t temp = std::sqrt(float_t(degrees[v]));
			if (temp == 0.0) norm_factor[v] = 0.0;
			else norm_factor[v] = 1.0 / temp;
		}, galois::loopname("NormCounting"));
	}

private:
	Graph *graph;
	bool act_; // whether to use activation function at the end
	bool norm_; // whether to normalize data
	bool bias_; // whether to add bias afterwards
	bool dropout_; // whether to use dropout at first
	const float dropout_rate_;
	float scale_;
	net_phase phase_;
	size_t x;
	size_t y;
	size_t z;
	vec_t out_temp;
	vec_t in_temp;
	vec_t trans_data; // y*x
	std::vector<unsigned> degrees;
	std::vector<float_t> norm_factor; // normalization constant based on graph structure
	std::vector<std::vector<unsigned> > dropout_mask;

	// Glorot & Bengio (AISTATS 2010) init
	inline void rand_init_matrix(size_t dim_x, size_t dim_y, vec_t &matrix) {
		auto init_range = sqrt(6.0/(dim_x + dim_y));
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
