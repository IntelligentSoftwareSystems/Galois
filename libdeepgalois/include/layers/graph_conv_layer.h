#pragma once
#include "layer.h"
#include "gtypes.h"

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
	void set_context(net_phase ctx) override { phase_ = ctx; }
	virtual void forward_propagation(const tensor_t &in_data, tensor_t &out_data);
	virtual void forward_propagation(const float_t *in_data, float_t *out_data);
	virtual void back_propagation(const tensor_t &in_data, const tensor_t &out_data, tensor_t &out_grad, tensor_t &in_grad);
	virtual void back_propagation(const float_t *in_data, const float_t *out_data, float_t *out_grad, float_t *in_grad);
	// user-defined aggregate function
	virtual void aggregate(Graph *g, const vec_t &in, tensor_t &out);
	// user-defined combine function
	virtual void combine(const vec_t &self, const vec_t &neighbors, const vec_t mat_v, const vec_t mat_u, vec_t &out) {
		vec_t a(out.size(), 0);
		vec_t b(out.size(), 0);
		mvmul(mat_v, self, a);
		mvmul(mat_u, neighbors, b); 
		vadd(a, b, out); // out = W*self + Q*neighbors
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
