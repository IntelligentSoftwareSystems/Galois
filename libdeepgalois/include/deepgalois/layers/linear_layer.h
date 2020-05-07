#pragma once
#include "layer.h"

namespace deepgalois {
class linear_layer : public layer {
public:
  linear_layer(unsigned level, float_t scale, float_t bias,
               std::vector<size_t> in_dims, std::vector<size_t> out_dims)
      : layer(level, in_dims, out_dims), scale_(scale), bias_(bias) {
    trainable_ = false;
  }
  linear_layer(unsigned level, std::vector<size_t> in_dim,
               std::vector<size_t> out_dim)
      : linear_layer(level, 1.0, 0.0, in_dim, out_dim) {}
  std::string layer_type() const override { return "linear"; }

  void forward_propagation(const tensor_t& in_data,
                           tensor_t& out_data) override {
    for (size_t sample = 0; sample < input_dims[0]; ++sample) {
      for (size_t i = 0; i < input_dims[1]; i++)
        out_data[sample][i] = scale_ * in_data[sample][i] + bias_;
    }
  }
  void back_propagation(const tensor_t& in_data, const tensor_t& out_data,
                        tensor_t& out_grad, tensor_t& in_grad) override {
    for (size_t sample = 0; sample < input_dims[0]; ++sample)
      for (size_t i = 0; i < input_dims[1]; i++)
        in_grad[sample][i] = out_grad[sample][i] * scale_;
  }

protected:
  float_t scale_, bias_;
};
} // namespace deepgalois
