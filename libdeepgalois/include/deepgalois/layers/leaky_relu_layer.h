#pragma once
#include "layer.h"

namespace deepgalois {
// Leaky ReLU Layer
class leaky_relu_layer : public layer {
public:
  leaky_relu_layer(unsigned level, float_t eps, dims_t in_dims, dims_t out_dims);
  leaky_relu_layer(unsigned level, dims_t in_dims, dims_t out_dims) :
    leaky_relu_layer(level, 0.0, in_dims, out_dims) {}
  ~leaky_relu_layer() {}
  std::string layer_type() const override { return std::string("leaky_relu"); }
  virtual void forward_propagation(const float_t* in_data, float_t* out_data);
  virtual void back_propagation(const float_t* in_data, const float_t* out_data,
                                float_t* out_grad, float_t* in_grad);
protected:
  float_t epsilon_;
  size_t n;
};
} // namespace
