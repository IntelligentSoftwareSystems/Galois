#include "layers/graph_conv_layer.h"
#include "aggregator.h"

void graph_conv_layer::aggregate(Graph *g, const vec_t &in, tensor_t &out) {
	update_all(g, in, out, true, norm_factor);
}

// 𝒉[𝑙] = σ(𝑊 * Σ(𝒉[𝑙-1]))
void graph_conv_layer::forward_propagation(const tensor_t &in_data, tensor_t &out_data) {
	// input: x*y; W: y*z; output: x*z
	// if y > z: mult W first to reduce the feature size for aggregation
	// else: aggregate first then mult W (not implemented yet)
	if (dropout_ && phase_ == net_phase::train) {
		galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
			dropout(scale_, dropout_rate_, in_data[i], dropout_mask[i], &in_temp[i*y]);
		}, galois::loopname("dropout"));
		matmul1D1D(x, z, y, in_temp, W, out_temp); // x*y; y*z; x*z
	} else matmul2D1D(z, in_data, W, out_temp); // x*y; y*z; x*z
	aggregate(graph, out_temp, out_data); // aggregate
	if (act_) {
		galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
			relu(out_data[i], out_data[i]);
		}, galois::loopname("relu"));
	}
}

void graph_conv_layer::forward_propagation(const float_t *in_data, float_t *out_data) { }

// 𝜕𝐸 / 𝜕𝑦[𝑙−1] = 𝜕𝐸 / 𝜕𝑦[𝑙] ∗ 𝑊 ^𝑇
void graph_conv_layer::back_propagation(const tensor_t &in_data, const tensor_t &out_data, tensor_t &out_grad, tensor_t &in_grad) {
	if (act_) {
		galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
			for (size_t j = 0; j < z; ++j) //TODO: use in_data or out_data?
				out_temp[i*z+j] = out_data[i][j] > float_t(0) ? out_grad[i][j] : float_t(0);
		}, galois::loopname("d_relu"));
	} else copy2D1D(out_grad, out_temp); // TODO: avoid copying
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

void graph_conv_layer::back_propagation(const float_t *in_data, const float_t *out_data, float_t *out_grad, float_t *in_grad) { }

