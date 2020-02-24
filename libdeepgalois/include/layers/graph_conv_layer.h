#pragma once
#include "layer.h"
#include "aggregator.h"

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
	graph_conv_layer(unsigned level, bool act, bool norm, bool bias, bool dropout,
		float dropout_rate, std::vector<size_t> in_dims, std::vector<size_t> out_dims);
	graph_conv_layer(unsigned level, std::vector<size_t> in_dims, std::vector<size_t> out_dims) :
		graph_conv_layer(level, false, true, false, true, 0.5, in_dims, out_dims) {}
	~graph_conv_layer() {}
	void init();
	std::string layer_type() const override { return std::string("graph_conv"); }
	void set_netphase(net_phase ctx) override { phase_ = ctx; }
	//virtual void forward_propagation(const vec_t &in_data, vec_t &out_data);
	//virtual void back_propagation(const vec_t &in_data, const vec_t &out_data, vec_t &out_grad, vec_t &in_grad);
	virtual void forward_propagation(const float_t *in_data, float_t *out_data);
	virtual void back_propagation(const float_t *in_data, const float_t *out_data, float_t *out_grad, float_t *in_grad);
	// user-defined aggregate function
#ifdef CPU_ONLY
	virtual void aggregate(size_t len, Graph &g, const float_t *in, float_t *out);
#else
	virtual void aggregate(size_t len, CSRGraph &g, const float_t *in, float_t *out);
#endif
	// user-defined combine function
	virtual void combine(const vec_t &self, const vec_t &neighbors, vec_t &out);

private:
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
	std::vector<unsigned>  dropout_mask; // x*y

	// Glorot & Bengio (AISTATS 2010)
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
