#include "deepgalois/layers/relu_layer.h"

namespace deepgalois {

#ifdef CPU_ONLY
// 𝑦[𝑙] = max(0, 𝑦[𝑙−1])
void relu_layer::forward_propagation(const float_t* in_data, float_t* out_data) {
  size_t n = input_dims[0] * input_dims[1];
  galois::do_all(galois::iterate((size_t)0, n), [&](const auto& i) {
    out_data[i] = std::max(in_data[i], (float_t)0);
  }, galois::chunk_size<64>(), galois::steal(), galois::loopname("relu_layer-fw"));
}

// 𝜕𝐿 / 𝜕𝑦[𝑙−1] = 0, 𝑖𝑓 (𝑦[𝑙] < 0)
//              = 𝜕𝐿 / 𝜕𝑦𝑙, 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒
void relu_layer::back_propagation(const float_t* in_data, const float_t* out_data,
                                  float_t* out_grad, float_t* in_grad) {
  size_t n = input_dims[0] * input_dims[1];
  galois::do_all(galois::iterate((size_t)0, n), [&](const auto& i) {
    in_grad[i] = out_data[i] > float_t(0) ? out_grad[i] : float_t(0);
  }, galois::chunk_size<64>(), galois::steal(), galois::loopname("relu_layer-bw"));
}
#endif

} // namespace
