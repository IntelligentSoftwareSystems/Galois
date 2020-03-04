#include "deepgalois/layers/graph_conv_layer.h"

namespace deepgalois {

graph_conv_layer::graph_conv_layer(unsigned level, bool act, bool norm,
                                   bool bias, bool dropout, float_t dropout_rate,
                                   std::vector<size_t> in_dims,
                                   std::vector<size_t> out_dims)
    : layer(level, in_dims, out_dims), act_(act), norm_(norm), bias_(bias),
      dropout_(dropout), dropout_rate_(dropout_rate) {
  assert(input_dims[0] == output_dims[0]); // num_vertices
  x          = input_dims[0];
  y          = input_dims[1];
  z          = output_dims[1];
  trainable_ = true;
  name_      = layer_type() + "_" + std::to_string(level);
#ifdef CPU_ONLY
  init();
#else
  init_gpu();
#endif
  assert(dropout_rate_ < 1.);
  scale_ = 1. / (1. - dropout_rate_);
}

#ifdef CPU_ONLY
void graph_conv_layer::aggregate(size_t len, Graph& g, const float_t* in, float_t* out) {
  deepgalois::update_all(len, g, in, out, true, context->norm_factor);
}

void graph_conv_layer::combine(size_t n, size_t len, const float_t* self, const float_t* neighbors, float_t* out) {
  float_t *a = new float_t[len];
  float_t *b = new float_t[len];
  mvmul(n, len, &Q[0], self, a);
  mvmul(n, len, &W[0], neighbors, b);
  deepgalois::math::vadd(len, a, b, out); // out = W*self + Q*neighbors
}

void graph_conv_layer::init() {
  rand_init_matrix(y, z, W); // randomly initialize trainable parameters
  // rand_init_matrix(y, z, Q);
  zero_init_matrix(y, z, layer::weight_grad);
  if (dropout_)
    dropout_mask = new unsigned[x * y];
  in_temp  = new float_t[x * y];
  out_temp = new float_t[x * z]; // same as pre_sup in original GCN code:
               // https://github.com/chenxuhao/gcn/blob/master/gcn/layers.py
  trans_data = new float_t[y * x]; // y*x
}

// 𝒉[𝑙] = σ(𝑊 * Σ(𝒉[𝑙-1]))
void graph_conv_layer::forward_propagation(const float_t* in_data, float_t* out_data) {
  // input: x*y; W: y*z; output: x*z
  // if y > z: mult W first to reduce the feature size for aggregation
  // else: aggregate first then mult W (not implemented yet)
  if (dropout_ && phase_ == deepgalois::net_phase::train) {
    galois::do_all(galois::iterate((size_t)0, x), [&](const auto& i) {
      deepgalois::math::dropout(y, scale_, dropout_rate_, &in_data[i * y],
                                &dropout_mask[i * y], &in_temp[i * y]);
    }, galois::loopname("dropout"));
    deepgalois::math::matmul1D1D(x, z, y, in_temp, &layer::W[0], out_temp); // x*y; y*z; x*z
  } else {
    deepgalois::math::matmul1D1D(x, z, y, in_data, &layer::W[0], out_temp); // x*y; y*z; x*z
  }

  // aggregate based on graph topology
  graph_conv_layer::aggregate(z, context->graph_cpu, out_temp, out_data);

  // run relu activation on output if specified
  if (act_) {
    galois::do_all(
        galois::iterate((size_t)0, x),
        [&](const auto& i) { deepgalois::math::relu(z, &out_data[i * z],
                                                    &out_data[i * z]); },
        galois::loopname("relu"));
  }
}

// 𝜕𝐸 / 𝜕𝑦[𝑙−1] = 𝜕𝐸 / 𝜕𝑦[𝑙] ∗ 𝑊 ^𝑇
void graph_conv_layer::back_propagation(const float_t* in_data,
                                        const float_t* out_data,
                                        float_t* out_grad, float_t* in_grad) {
  if (act_) {
    // note; assumption here is that out_grad contains 1s or 0s via relu?
    galois::do_all(galois::iterate((size_t)0, x),
      [&](const auto& i) {
        for (size_t j = 0; j < z; ++j) // TODO: use in_data or out_data?
          // check if original data greater than 0; if so keep grad
          out_temp[i * z + j] = out_data[i * z + j] > float_t(0)
                                ? out_grad[i * z + j] : float_t(0);
      }, galois::loopname("d_relu"));
  } else {
    deepgalois::math::copy1D1D(x * z, out_grad, out_temp); // TODO: avoid copying
  }

  // at this point, out_temp has the derivative of data from last step to
  // use for both updating gradients for features and gradients for weights

  // this calculates gradients for the node predictions
  if (level_ != 0) { // no need to calculate in_grad for the first layer
    vec_t trans_W(z * y);
    // derivative of matmul needs transposed matrix
    deepgalois::math::transpose(y, z, layer::W, trans_W);
    deepgalois::math::matmul1D1D(x, y, z, out_temp, &trans_W[0], in_temp); // x*z; z*y -> x*y
    // sgemm_cpu(x, y, z, 1.0, out_temp, trans_W, 0.0, in_temp); // x*z; z*y ->
    // x*y NOTE: since graph is symmetric, the derivative is the same
    deepgalois::update_all(y, context->graph_cpu, in_temp, in_grad, true,
                           context->norm_factor); // x*x; x*y -> x*y
    if (dropout_) {
      galois::do_all(galois::iterate((size_t)0, x),
        [&](const auto& i) {
          deepgalois::math::d_dropout(y, scale_, &in_grad[i * y],
                    &dropout_mask[i * y], &in_grad[i * y]);
        }, galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
        galois::loopname("d_dropout"));
    }
  }

  // calculate weight gradients by using previous layer's transpose multiplied
  // by gradients from last back prop step
  deepgalois::math::transpose(x, y, in_data, trans_data);                       // y*x
  // updates THIS layer's weight gradients to update them
  deepgalois::math::matmul1D1D(y, z, x, trans_data, out_temp, &layer::weight_grad[0]); // y*x; x*z; y*z
}

#endif

} // namespace
