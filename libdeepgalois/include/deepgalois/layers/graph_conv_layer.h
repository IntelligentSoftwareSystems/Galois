#pragma once
#include "layer.h"
#include "deepgalois/layers/aggregator.h"
#ifdef GALOIS_USE_DIST
#include "deepgalois/layers/GraphConvSyncStructures.h"
#endif

/**
 * GraphConv Layer; based on DGL implementation + follows TinyDNN layer
 * convention 
 * https://docs.dgl.ai/en/0.4.x/_modules/dgl/nn/pytorch/conv/graphconv.html
 *
 *   Parameters
 *   ----------
 *   x: int, number of samples.
 *   y: int, Input feature size.
 *   z: int, Output feature size.
 *   dropout: bool, optional, if True, a dropout operation is applied before
 *   other operations.
 *   norm : bool, optional, if True, the normalizer :math:`c_{ij}` is applied.
 *          Default: ``True``.
 *   bias : bool, optional, if True, adds a learnable bias to the output.
 *          Default: ``False``.
 *   activation: default false
 */
namespace deepgalois {
class graph_conv_layer : public layer {
public:
  graph_conv_layer(unsigned level, bool act, bool norm, bool bias,
                   bool dropout, float_t dropout_rate,
                   std::vector<size_t> in_dims, std::vector<size_t> out_dims)
    : layer(level, in_dims, out_dims), act_(act), norm_(norm), bias_(bias),
      dropout_(dropout), dropout_rate_(dropout_rate) {
    assert(input_dims[0] == output_dims[0]); // num_vertices
    trainable_ = true;
    name_      = layer_type() + "_" + std::to_string(level);
    assert(dropout_rate_ >= 0. && dropout_rate_ < 1.);
    scale_ = 1. / (1. - dropout_rate_);
  }
  graph_conv_layer(unsigned level, std::vector<size_t> in_dims,
                   std::vector<size_t> out_dims)
      : graph_conv_layer(level, false, true, false, true, 0.5, in_dims, out_dims) {}
  ~graph_conv_layer() {}
  void malloc_and_init();
  std::string layer_type() const override { return std::string("graph_conv"); }
  virtual acc_t get_weight_decay_loss();
  //! Uses weights contained in this layer to update in_data (results from previous)
  //! and save result to out_data
  virtual void forward_propagation(const float_t* in_data, float_t* out_data);
  //! Uses gradients from layer after this one to update both own weight gradients
  //! as well as gradients for the features (in_grad)
  virtual void back_propagation(const float_t* in_data, const float_t* out_data,
                                float_t* out_grad, float_t* in_grad);
  // user-defined aggregate function
#ifdef CPU_ONLY
  virtual void aggregate(size_t len, Graph& g, const float_t* in, float_t* out);
  void d_aggregate(size_t len, Graph& g, const float_t* in, float_t* out);
#else
  virtual void aggregate(size_t len, CSRGraph& g, const float_t* in, float_t* out);
  void d_aggregate(size_t len, CSRGraph& g, const float_t* in, float_t* out);
#endif
  // user-defined combine function
  virtual void combine(size_t dim_x, size_t dim_y, const float_t* self, const float_t* neighbors, float_t* out);

private:
  bool act_;     // whether to use activation function at the end
  bool norm_;    // whether to normalize data
  bool bias_;    // whether to add bias afterwards
  bool dropout_; // whether to use dropout at first
  const float_t dropout_rate_;
  float_t scale_;
  float_t* out_temp; //!< intermediate data temporary
  float_t* in_temp;
  float_t* in_temp1;
  float_t* trans_data;    // y*x
  mask_t* dropout_mask; // x*y

  // Glorot & Bengio (AISTATS 2010)
  inline void rand_init_matrix(size_t dim_x, size_t dim_y, vec_t& matrix, unsigned seed=1);
  inline void zero_init_matrix(size_t dim_x, size_t dim_y, vec_t& matrix);
};

} // namespace
