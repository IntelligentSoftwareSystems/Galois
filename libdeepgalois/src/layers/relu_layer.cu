#include "deepgalois/layers/relu_layer.h"

namespace deepgalois {

// 𝑦[𝑙] = max(0, 𝑦[𝑙−1])
void relu_layer::forward_propagation(const float_t* in_data, float_t* out_data) {
  const size_t count = input_dims[0] * input_dims[1];
  relu_gpu(count, in_data, out_data);
}

// 𝜕𝐿 / 𝜕𝑦[𝑙−1] = 0, 𝑖𝑓 (𝑦[𝑙] < 0)
//              = 𝜕𝐿 / 𝜕𝑦𝑙, 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒
void relu_layer::back_propagation(const float_t* in_data, const float_t* out_data,
                                  float_t* out_grad, float_t* in_grad) {
  const size_t count = input_dims[0] * input_dims[1];
  d_relu_gpu(count, out_grad, in_data, in_grad);
}

} // namespace
