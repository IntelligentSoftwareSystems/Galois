#include "deepgalois/layers/graph_conv_layer.h"
#include "deepgalois/math_functions.hh"
#include "deepgalois/utils.h"

static galois::DynamicBitSet bitset_conv;

#include "deepgalois/layers/GraphConvSyncStructures.h"
#include "deepgalois/layers/GradientSyncStructs.h"

namespace deepgalois {
#include "gat_fw.h"

//! Set this to let sync struct know where to get data from
float_t* _dataToSync = nullptr;
//! Set this to let sync struct know the size of the vector to use during
//! sync
long unsigned _syncVectorSize = 0;

inline void graph_conv_layer::rand_init_matrix(size_t dim_x, size_t dim_y,
                                               vec_t& matrix, unsigned seed) {
  auto init_range = sqrt(6.0 / (dim_x + dim_y));
  std::default_random_engine rng(seed);
  std::uniform_real_distribution<float_t> dist(-init_range, init_range);
  matrix.resize(dim_x * dim_y);
  for (size_t i = 0; i < dim_x; ++i) {
    for (size_t j = 0; j < dim_y; ++j)
      matrix[i * dim_y + j] = dist(rng);
  }
}

inline void graph_conv_layer::zero_init_matrix(size_t dim_x, size_t dim_y,
                                               vec_t& matrix) {
  matrix.resize(dim_x * dim_y);
  for (size_t i = 0; i < dim_x; ++i) {
    for (size_t j = 0; j < dim_y; ++j)
      matrix[i * dim_y + j] = 0;
  }
}

void graph_conv_layer::malloc_and_init() {
  size_t x = input_dims[0];
  size_t y = input_dims[1];
  size_t z = output_dims[1];

  galois::gInfo("conv bitset size is going to be ", x);
  bitset_conv.resize(x);

  // setup gluon
  layer::gradientGraph =
      new deepgalois::GluonGradients(layer::weight_grad, y * z);
  layer::syncSub =
      new galois::graphs::GluonSubstrate<deepgalois::GluonGradients>(
          *layer::gradientGraph, layer::gradientGraph->myHostID(),
          layer::gradientGraph->numHosts(), false);
  galois::gInfo("gradient bitset size is going to be ", y * z, " ", y, " ", z);

  // make sure seed consistent across all hosts for weight matrix
  rand_init_matrix(y, z, W, 1);
  //rand_init_matrix(y, z, Q, 1); // for GraphSAGE

  zero_init_matrix(y, z, layer::weight_grad);

#ifdef USE_GAT
  // alpha is only used for GAT
  rand_init_matrix(z, 1, alpha_l, 1);
  rand_init_matrix(z, 1, alpha_r, 1);
  alpha_lgrad.resize(2*z);
  alpha_rgrad.resize(2*z);
  std::fill(alpha_lgrad.begin(), alpha_lgrad.end(), 0);
  std::fill(alpha_rgrad.begin(), alpha_rgrad.end(), 0);
  auto ne = graph_cpu->sizeEdges(); // number of edges
  scores.resize(ne); // a score for each edge
  temp_scores.resize(ne);
  scores_grad.resize(ne);
  norm_scores.resize(ne);
  norm_scores_grad.resize(ne);
  epsilon = 0.2; // LeakyReLU angle of negative slope
#endif

  if (dropout_)
    dropout_mask = new mask_t[x * y];
  in_temp    = new float_t[x * y];
  out_temp   = new float_t[x * z];
  trans_data = new float_t[y * x]; // y*x
  if (y <= z)
    in_temp1 = new float_t[x * y];
}

namespace {
void set_conv_bitset() {
  // bitset setting
  galois::do_all(
      galois::iterate((size_t)0, bitset_conv.size()),
      [&](size_t node_id) {
        bool set_true = false;
        // check for non-zeros; the moment one is found, set true becomes true
        // and we break out of the loop
        for (size_t i = 0; i < deepgalois::_syncVectorSize; i++) {
          auto val =
              deepgalois::_dataToSync[node_id * deepgalois::_syncVectorSize +
                                      i];
          if (val != 0) {
            set_true = true;
            break;
          }
        }

        if (set_true) {
          bitset_conv.set(node_id);
        }
      },
      galois::loopname("BitsetGraphConv"), galois::no_stats());
}

} // end anonymous namespace

void graph_conv_layer::combine(size_t n, size_t len, const float_t* self,
                               const float_t* neighbors, float_t* out) {
  float_t* a = new float_t[len];
  float_t* b = new float_t[len];
  math::mvmul(CblasNoTrans, n, len, 1.0, &Q[0], self, 0.0, a);
  math::mvmul(CblasNoTrans, n, len, 1.0, &W[0], neighbors, 0.0, b);
  math::vadd_cpu(len, a, b, out); // out = W*self + Q*neighbors
}

#ifndef USE_GAT
// aggregate based on graph topology
void graph_conv_layer::aggregate(size_t len, Graph& g, const float_t* in,
                                 float_t* out) {
  galois::StatTimer aggregate_timer("AggregateTime");
  aggregate_timer.start();
  // normalization constant based on graph structure
#ifdef USE_MKL
  update_all_csrmm(len, g, in, out, norm_, norm_consts);
#else
  update_all(len, g, in, out, norm_, norm_consts);
#endif
  aggregate_timer.stop();
}

// since graph is symmetric, the derivative is the same
void graph_conv_layer::d_aggregate(size_t len, Graph& g, const float_t* in,
                                   float_t* out) {
  galois::StatTimer aggregate_timer("AggregateDerivativeTime");
  aggregate_timer.start();
#ifdef USE_MKL
  update_all_csrmm(len, g, in, out, norm_, norm_consts); // x*x; x*z -> x*z
#else
  update_all(len, g, in, out, norm_, norm_consts); // x*x; x*z -> x*z
#endif
  aggregate_timer.stop();
}

// 𝒉[𝑙] = σ(𝑊 * Σ(𝒉[𝑙-1]))
void graph_conv_layer::forward_propagation(const float_t* in_data,
                                           float_t* out_data) {
  galois::StatTimer conv_timer("GraphConvForward");
  conv_timer.start();
  size_t x = input_dims[0];
  size_t y = input_dims[1];
  size_t z = output_dims[1];
  galois::gPrint("forward ", x, " ", y, " ", z, "\n");

  galois::StatTimer drop_timer("GraphConvForwardDropout");
  drop_timer.start();
  // input: x*y; W: y*z; output: x*z
  // if y > z: mult W first to reduce the feature size for aggregation
  // else: aggregate first then mult W
  if (dropout_ && phase_ == net_phase::train) {
    math::dropout_cpu(x, y, scale_, dropout_rate_, in_data, dropout_mask,
                      in_temp);
  } else {
    math::copy_cpu(x * y, in_data, in_temp);
  }
  drop_timer.stop();

  galois::StatTimer compute_timer("GraphConvForwardCompute");
  compute_timer.start();
  if (y > z) {
    math::sgemm_cpu(CblasNoTrans, CblasNoTrans, x, z, y, 1.0, in_temp,
                    &layer::W[0], 0.0, out_temp);
    aggregate(z, *graph_cpu, out_temp, out_data);
  } else {
    aggregate(y, *graph_cpu, in_temp, in_temp1);
    math::sgemm_cpu(CblasNoTrans, CblasNoTrans, x, z, y, 1.0, in_temp1,
                    &layer::W[0], 0.0, out_data);
  }
  compute_timer.stop();

  // TODO sync of out_data required here
  // TODO how to do this for the sampled case?
  deepgalois::_syncVectorSize = z;
  deepgalois::_dataToSync     = out_data;
  set_conv_bitset();

  galois::gPrint("forward ", x, " ", y, " ", z, " sync calling\n");
  layer::context->getSyncSubstrate()
      ->sync<writeAny, readAny, GraphConvSync, Bitset_conv>("GraphConvForward");

  // run relu activation on output if specified
  galois::StatTimer relu_timer("GraphConvForwardRelu");
  relu_timer.start();
  if (act_)
    math::relu_cpu(x * z, out_data, out_data);
  relu_timer.stop();

  conv_timer.stop();
}

// 𝜕𝐸 / 𝜕𝑦[𝑙−1] = 𝜕𝐸 / 𝜕𝑦[𝑙] ∗ 𝑊 ^𝑇
void graph_conv_layer::back_propagation(const float_t* in_data,
                                        const float_t* out_data,
                                        float_t* out_grad, float_t* in_grad) {
  galois::StatTimer conv_timer("GraphConvBackward");
  conv_timer.start();
  size_t x = input_dims[0];
  size_t y = input_dims[1];
  size_t z = output_dims[1];
  // note; assumption here is that out_grad contains 1s or 0s via relu?
  galois::StatTimer relu_timer("GraphConvBackwardRelu");
  relu_timer.start();
  if (act_)
    math::d_relu_cpu(x * z, out_grad, out_data, out_grad);
  relu_timer.stop();
  // else math::copy_cpu(x * z, out_grad, out_temp); // TODO: avoid copying

  galois::StatTimer compute_timer("GraphConvBackwardCompute");
  compute_timer.start();
  if (y > z) {
    d_aggregate(z, *graph_cpu, out_grad, out_temp);
    // at this point, out_temp has the derivative of data from last step to
    // use for both updating gradients for features and gradients for weights
    // this calculates gradients for the node predictions
    if (level_ != 0) {// no need to calculate in_grad for the first layer
      // derivative of matmul needs transposed matrix
      math::sgemm_cpu(CblasNoTrans, CblasTrans, x, y, z, 1.0, out_temp, &W[0],
                      0.0, in_grad); // x*z; z*y -> x*y
    }
    // calculate weight gradients using input data; multiplied by gradients from
    // last back prop step
    math::sgemm_cpu(CblasTrans, CblasNoTrans, y, z, x, 1.0, in_data, out_temp,
                    0.0, &layer::weight_grad[0]); // y*x; x*z; y*z
  } else {
    if (level_ != 0) {
      math::sgemm_cpu(CblasNoTrans, CblasTrans, x, y, z, 1.0, out_grad, &W[0],
                      0.0, in_temp);
      d_aggregate(y, *graph_cpu, in_temp, in_grad);
    }
    math::sgemm_cpu(CblasTrans, CblasNoTrans, y, z, x, 1.0, in_data, out_grad,
                    0.0, &layer::weight_grad[0]);
  }
  compute_timer.stop();

  // sync agg
  //galois::gPrint(header, "x is ", x, " y is ", y,  " z is ", z, "\n");
  if (level_ != 0) {
    deepgalois::_syncVectorSize = y;
    deepgalois::_dataToSync     = in_grad;
    set_conv_bitset();
    layer::context->getSyncSubstrate()
        ->sync<writeAny, readAny, GraphConvSync, Bitset_conv>(
            //->sync<writeAny, readAny, GraphConvSync>(
            "GraphConvBackward");
  }
  galois::StatTimer drop_timer("GraphConvBackwardDropout");
  drop_timer.start();
  if (level_ != 0 && dropout_)
    math::d_dropout_cpu(x, y, scale_, in_grad, dropout_mask, in_grad);
  drop_timer.stop();

  deepgalois::_syncVectorSize = z;
  deepgalois::_dataToSync     = &layer::weight_grad[0];
  unsigned host_num = galois::runtime::getSystemNetworkInterface().Num;
  layer::syncSub->sync<writeAny, readAny, GradientSync>("Gradients");
  galois::do_all(
    galois::iterate((size_t)0, (size_t)z),
    [&] (size_t i) {
      //galois::gPrint("before ", i, " ", layer::weight_grad[i], "\n");
      layer::weight_grad[i] /= host_num;
      //galois::gPrint("after ", i, " ", layer::weight_grad[i], "\n");
    },
    galois::loopname("sync post process")
  );

  galois::gDebug("[", layer::gradientGraph->myHostID(), "] Sync done");
  conv_timer.stop();
}
#endif

acc_t graph_conv_layer::get_weight_decay_loss() {
  return math::l2_norm(input_dims[1] * output_dims[1], &layer::W[0]);
}

} // namespace deepgalois
