#pragma once
#include "layer.h"

namespace deepgalois {
// element-wise add N vectors ```y_i = x0_i + x1_i + ... + xnum_i```
class elementwise_add_layer : public layer {
public:
  elementwise_add_layer(unsigned level, std::vector<size_t> in_dim,
                        std::vector<size_t> out_dim)
      : layer(level, in_dim, out_dim) {
    trainable_ = false;
  }
  std::string layer_type() const override {
    return std::string("elementwise_add");
  }
  void forward_propagation(const tensor_t& in_data,
                           tensor_t& out_data) override {
    for (size_t sample = 0; sample < in_data.size(); ++sample) {
      for (size_t j = 0; j < in_data[0].size(); j++)
        out_data[sample][j] = in_data[sample][j];
    }
  }
  void back_propagation(const tensor_t& in_data, const tensor_t& out_data,
                        tensor_t& out_grad, tensor_t& in_grad) override {
    in_grad = out_grad;
  }
};
} // namespace
