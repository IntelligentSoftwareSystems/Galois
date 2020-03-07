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
  init();
  assert(dropout_rate_ < 1.);
  scale_ = 1. / (1. - dropout_rate_);
}

#ifdef CPU_ONLY
void graph_conv_layer::aggregate(size_t len, Graph& g, const float_t* in, float_t* out) {
  deepgalois::update_all(len, g, in, out, norm_, norm_factor);
}

void graph_conv_layer::combine(size_t n, size_t len, const float_t* self, const float_t* neighbors, float_t* out) {
  float_t *a = new float_t[len];
  float_t *b = new float_t[len];
  mvmul(n, len, &Q[0], self, a);
  mvmul(n, len, &W[0], neighbors, b);
  deepgalois::math::vadd_cpu(len, a, b, out); // out = W*self + Q*neighbors
}

void graph_conv_layer::init() {
  rand_init_matrix(y, z, W); // randomly initialize trainable parameters
  // rand_init_matrix(y, z, Q);
  zero_init_matrix(y, z, layer::weight_grad);

#ifdef GALOIS_USE_DIST
  // setup gluon
  layer::gradientGraph = new deepgalois::GluonGradients(layer::weight_grad,
                                                        y * z);
  layer::syncSub =
    new galois::graphs::GluonSubstrate<deepgalois::GluonGradients>(
      *layer::gradientGraph, layer::gradientGraph->myHostID(),
      layer::gradientGraph->numHosts(), false);
#endif

  if (dropout_) dropout_mask = new unsigned[x * y];
  in_temp  = new float_t[x * y];
  out_temp = new float_t[x * z];
  trans_data = new float_t[y * x]; // y*x
}

// 𝒉[𝑙] = σ(𝑊 * Σ(𝒉[𝑙-1]))
void graph_conv_layer::forward_propagation(const float_t* in_data, float_t* out_data) {
  // input: x*y; W: y*z; output: x*z
  // if y > z: mult W first to reduce the feature size for aggregation
  // else: aggregate first then mult W (not implemented yet)
  if (dropout_ && phase_ == deepgalois::net_phase::train) {
    deepgalois::math::dropout_cpu(x*y, scale_, dropout_rate_, in_data, dropout_mask, in_temp);
    deepgalois::math::sgemm_cpu(CblasNoTrans, CblasNoTrans, x, z, y, 1.0, in_temp, &layer::W[0], 0.0, out_temp);
  } else deepgalois::math::sgemm_cpu(CblasNoTrans, CblasNoTrans, x, z, y, 1.0, in_data, &layer::W[0], 0.0, out_temp);

  // aggregate based on graph topology
  graph_conv_layer::aggregate(z, context->graph_cpu, out_temp, out_data);

  // run relu activation on output if specified
  if (act_) deepgalois::math::relu_cpu(x*z, out_data, out_data);
}

// 𝜕𝐸 / 𝜕𝑦[𝑙−1] = 𝜕𝐸 / 𝜕𝑦[𝑙] ∗ 𝑊 ^𝑇
void graph_conv_layer::back_propagation(const float_t* in_data,
                                        const float_t* out_data,
                                        float_t* out_grad, float_t* in_grad) {
  // note; assumption here is that out_grad contains 1s or 0s via relu?
  if (act_) deepgalois::math::d_relu_cpu(x*z, out_grad, out_data, out_grad);
  //else deepgalois::math::copy_cpu(x * z, out_grad, out_temp); // TODO: avoid copying

  // x*y NOTE: since graph is symmetric, the derivative is the same
  deepgalois::update_all(z, context->graph_cpu, out_grad, out_temp, norm_, norm_factor); // x*x; x*z -> x*z

  // at this point, out_temp has the derivative of data from last step to
  // use for both updating gradients for features and gradients for weights
  // this calculates gradients for the node predictions
  if (level_ != 0) { // no need to calculate in_grad for the first layer
    // derivative of matmul needs transposed matrix
    deepgalois::math::sgemm_cpu(CblasNoTrans, CblasTrans, x, y, z, 1.0,
                                out_temp, &W[0], 0.0, in_grad); // x*z; z*y -> x*y
    if (dropout_) {
      deepgalois::math::d_dropout_cpu(x*y, scale_, in_grad, dropout_mask,
                                      in_grad);
    }
  }

  // calculate weight gradients using input data
  // multiplied by gradients from last back prop step
  deepgalois::math::sgemm_cpu(CblasTrans, CblasNoTrans, y, z, x, 1.0, in_data,
                              out_temp, 0.0, &layer::weight_grad[0]); // y*x; x*z; y*z
  layer::syncSub->sync<writeAny, readAny, GradientSync>("GradientSync");
  //galois::gInfo("[", layer::gradientGraph->myHostID(), "] Sync done");
}
#endif
} // namespace

