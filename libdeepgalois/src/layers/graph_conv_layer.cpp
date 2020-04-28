#include "deepgalois/layers/graph_conv_layer.h"
#include "deepgalois/utils.h"

namespace deepgalois {

graph_conv_layer::graph_conv_layer(unsigned level, bool act, bool norm,
                                   bool bias, bool dropout, float_t dropout_rate,
                                   std::vector<size_t> in_dims,
                                   std::vector<size_t> out_dims)
    : layer(level, in_dims, out_dims), act_(act), norm_(norm), bias_(bias),
      dropout_(dropout), dropout_rate_(dropout_rate) {
  assert(input_dims[0] == output_dims[0]); // num_vertices
  trainable_ = true;
  name_      = layer_type() + "_" + std::to_string(level);
  assert(dropout_rate_ < 1.);
  scale_ = 1. / (1. - dropout_rate_);
}

inline void graph_conv_layer::rand_init_matrix(size_t dim_x, size_t dim_y, vec_t& matrix, unsigned seed) {
  auto init_range = sqrt(6.0 / (dim_x + dim_y));
  std::default_random_engine rng(seed);
  std::uniform_real_distribution<float_t> dist(-init_range, init_range);
  matrix.resize(dim_x * dim_y);
  for (size_t i = 0; i < dim_x; ++i) {
    for (size_t j = 0; j < dim_y; ++j)
      matrix[i * dim_y + j] = dist(rng);
  }
}

inline void graph_conv_layer::zero_init_matrix(size_t dim_x, size_t dim_y, vec_t& matrix) {
  matrix.resize(dim_x * dim_y);
  for (size_t i = 0; i < dim_x; ++i) {
    for (size_t j = 0; j < dim_y; ++j)
      matrix[i * dim_y + j] = 0;
  }
}

#ifdef CPU_ONLY
// aggregate based on graph topology
void graph_conv_layer::aggregate(size_t len, Graph& g, const float_t* in, float_t* out) {
  // normalization constant based on graph structure
  float_t* norm_consts = context->get_norm_factor_ptr();
  update_all(len, g, in, out, norm_, norm_consts);
}

// since graph is symmetric, the derivative is the same
void graph_conv_layer::d_aggregate(size_t len, Graph& g, const float_t* in, float_t* out) {
  float_t* norm_consts = context->get_norm_factor_ptr();
  update_all(len, g, in, out, norm_, norm_consts); // x*x; x*z -> x*z
}

void graph_conv_layer::combine(size_t n, size_t len, const float_t* self, const float_t* neighbors, float_t* out) {
  float_t *a = new float_t[len];
  float_t *b = new float_t[len];
  mvmul(n, len, &Q[0], self, a);
  mvmul(n, len, &W[0], neighbors, b);
  math::vadd_cpu(len, a, b, out); // out = W*self + Q*neighbors
}

void graph_conv_layer::malloc_and_init() {
  size_t x = input_dims[0];
  size_t y = input_dims[1];
  size_t z = output_dims[1];
#ifdef GALOIS_USE_DIST
  // setup gluon
  layer::gradientGraph = new deepgalois::GluonGradients(layer::weight_grad,
                                                        y * z);
  layer::syncSub =
    new galois::graphs::GluonSubstrate<deepgalois::GluonGradients>(
      *layer::gradientGraph, layer::gradientGraph->myHostID(),
      layer::gradientGraph->numHosts(), false);
#endif

#ifdef GALOIS_USE_DIST
  // make sure seed consistent across all hosts for weight matrix
  rand_init_matrix(y, z, W, 1);
#else
  rand_init_matrix(y, z, W);
#endif

  // rand_init_matrix(y, z, Q);
  zero_init_matrix(y, z, layer::weight_grad);

  if (dropout_) dropout_mask = new unsigned[x * y];
  in_temp  = new float_t[x * y];
  out_temp = new float_t[x * z];
  trans_data = new float_t[y * x]; // y*x
  if (y <= z) in_temp1 = new float_t[x * y];
}

// 𝒉[𝑙] = σ(𝑊 * Σ(𝒉[𝑙-1]))
void graph_conv_layer::forward_propagation(const float_t* in_data, float_t* out_data) {
  size_t x = input_dims[0];
  size_t y = input_dims[1];
  size_t z = output_dims[1];
  //std::cout << "x=" << x << ", y=" << y << ", z=" << z << "\n";

  // input: x*y; W: y*z; output: x*z
  // if y > z: mult W first to reduce the feature size for aggregation
  // else: aggregate first then mult W
  if (dropout_ && phase_ == net_phase::train)
    math::dropout_cpu(x*y, scale_, dropout_rate_, in_data, dropout_mask, in_temp);
  else math::copy_cpu(x*y, in_data, in_temp); 

  if (y > z) {
    math::sgemm_cpu(CblasNoTrans, CblasNoTrans, x, z, y, 1.0, in_temp, &layer::W[0], 0.0, out_temp);
    aggregate(z, *graph_cpu, out_temp, out_data);
  } else {
    aggregate(y, *graph_cpu, in_temp, in_temp1);
    math::sgemm_cpu(CblasNoTrans, CblasNoTrans, x, z, y, 1.0, in_temp1, &layer::W[0], 0.0, out_data);
  }
#ifdef GALOIS_USE_DIST
  // TODO sync of out_data required here
  deepgalois::_syncVectorSize = z;
  deepgalois::_dataToSync = out_data;
  layer::context->getSyncSubstrate()->sync<writeAny, readAny, GraphConvSync>("AggSync");
#endif
  // run relu activation on output if specified
  if (act_) math::relu_cpu(x*z, out_data, out_data);
}

// 𝜕𝐸 / 𝜕𝑦[𝑙−1] = 𝜕𝐸 / 𝜕𝑦[𝑙] ∗ 𝑊 ^𝑇
void graph_conv_layer::back_propagation(const float_t* in_data,
                                        const float_t* out_data,
                                        float_t* out_grad, float_t* in_grad) {
  size_t x = input_dims[0];
  size_t y = input_dims[1];
  size_t z = output_dims[1];
  // note; assumption here is that out_grad contains 1s or 0s via relu?
  if (act_) math::d_relu_cpu(x*z, out_grad, out_data, out_grad);
  //else math::copy_cpu(x * z, out_grad, out_temp); // TODO: avoid copying

  if (y > z) {
    d_aggregate(z, *graph_cpu, out_grad, out_temp);
    // at this point, out_temp has the derivative of data from last step to
    // use for both updating gradients for features and gradients for weights
    // this calculates gradients for the node predictions
    if (level_ != 0) // no need to calculate in_grad for the first layer
      // derivative of matmul needs transposed matrix
      math::sgemm_cpu(CblasNoTrans, CblasTrans, x, y, z, 1.0, out_temp, &W[0], 0.0, in_grad); // x*z; z*y -> x*y
    // calculate weight gradients using input data; multiplied by gradients from last back prop step
    math::sgemm_cpu(CblasTrans, CblasNoTrans, y, z, x, 1.0, in_data, out_temp, 0.0, &layer::weight_grad[0]); // y*x; x*z; y*z
  } else {
    if (level_ != 0) {
      math::sgemm_cpu(CblasNoTrans, CblasTrans, x, y, z, 1.0, out_grad, &W[0], 0.0, in_temp);
      d_aggregate(y, *graph_cpu, in_temp, in_grad);
    }
    math::sgemm_cpu(CblasTrans, CblasNoTrans, y, z, x, 1.0, in_data, out_grad, 0.0, &layer::weight_grad[0]);
  }

#ifdef GALOIS_USE_DIST
  // sync agg
  deepgalois::_syncVectorSize = z;
  deepgalois::_dataToSync = out_temp;
  layer::context->getSyncSubstrate()->sync<writeAny, readAny, GraphConvSync>("AggSyncBack");
#endif

  if (level_ != 0 && dropout_)
    math::d_dropout_cpu(x*y, scale_, in_grad, dropout_mask, in_grad);

#ifdef GALOIS_USE_DIST
  layer::syncSub->sync<writeAny, readAny, GradientSync>("GradientSync");
  //galois::gInfo("[", layer::gradientGraph->myHostID(), "] Sync done");
#endif
}

acc_t graph_conv_layer::get_weight_decay_loss() {
  return math::l2_norm(input_dims[1]*output_dims[1], &layer::W[0]);
}

#endif // end if CPU_ONLY
} // namespace

