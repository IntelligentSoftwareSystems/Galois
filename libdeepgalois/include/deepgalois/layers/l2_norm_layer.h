#pragma once
#include "layer.h"

namespace deepgalois {
// L2 Normalization Layer
class l2_norm_layer : public layer {
public:
  l2_norm_layer(unsigned level, float_t eps, float_t scale, dims_t in_dims,
                dims_t out_dims)
      : layer(level, in_dims, out_dims), epsilon_(eps), scale_(scale) {
    assert(input_dims[0] == output_dims[0]); // num_vertices
    trainable_ = false;
    name_      = layer_type() + "_" + std::to_string(level);
  }
  l2_norm_layer(unsigned level, dims_t in_dims, dims_t out_dims)
      : l2_norm_layer(level, 1e-12, 20, in_dims, out_dims) {}
  ~l2_norm_layer() {}
  std::string layer_type() const override { return std::string("l2_norm"); }
  virtual void forward_propagation(const float_t* in_data, float_t* out_data);
  virtual void back_propagation(const float_t* in_data, const float_t* out_data,
                                float_t* out_grad, float_t* in_grad);

protected:
  float_t epsilon_;
  float_t scale_;
};

} // namespace deepgalois
