#pragma once
#include "layer.h"

// ReLU Layer
class relu_layer : public layer {
public:
  relu_layer(unsigned level, std::vector<size_t> in_dims,
             std::vector<size_t> out_dims)
      : layer(level, in_dims, out_dims) {
    trainable_ = false;
  }
  std::string layer_type() const override { return std::string("relu"); }
  // 𝑦[𝑙] = max(0, 𝑦[𝑙−1])
  void forward_propagation(const tensor_t& in_data,
                           tensor_t& out_data) override {
    galois::do_all(
        galois::iterate((size_t)0, input_dims[0]),
        [&](const auto& i) {
          for (size_t j = 0; j < input_dims[1]; ++j)
            out_data[i][j] =
                std::max(in_data[i][j], (float_t)0) +
                negative_slope * std::min(in_data[i][j], (float_t)0);
        },
        galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
        galois::loopname("relu_layer-fw"));
  }
  // 𝜕𝐿 / 𝜕𝑦[𝑙−1] = 0, 𝑖𝑓 (𝑦[𝑙] < 0)
  //              = 𝜕𝐿 / 𝜕𝑦𝑙 , 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒
  void back_propagation(const tensor_t& in_data, const tensor_t& out_data,
                        tensor_t& out_grad, tensor_t& in_grad) override {}
};
