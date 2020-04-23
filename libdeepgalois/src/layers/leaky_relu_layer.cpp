#include "deepgalois/layers/leaky_relu_layer.h"

namespace deepgalois {

leaky_relu_layer::leaky_relu_layer(unsigned level, float_t eps,
                                   dims_t in_dims, dims_t out_dims)
    : layer(level, in_dims, out_dims), epsilon_(eps) {
  assert(input_dims[0] == output_dims[0]); // num_vertices
  trainable_ = false;
  n = input_dims[0] * input_dims[1];
  name_ = layer_type() + "_" + std::to_string(level);
}

#ifdef CPU_ONLY
// 𝑦[𝑙] = 𝑦[𝑙−1] > 0 ? 𝑦[𝑙−1]) : 𝑦[𝑙−1] * ε 
void leaky_relu_layer::forward_propagation(const float_t* in_data, float_t* out_data) {
  galois::do_all(galois::iterate((size_t)0, n), [&](const auto& i) {
    out_data[i] = in_data[i] > (float_t)0 ? in_data[i] : epsilon_ * in_data[i];
  }, galois::chunk_size<64>(), galois::steal(), galois::loopname("leaky_relu_layer-fw"));
}

// 𝜕𝐿 / 𝜕𝑦[𝑙−1] = 𝜕𝐿 / 𝜕𝑦𝑙 * ε,   𝑖𝑓 (𝑦[𝑙] ≤ 0)
//              = 𝜕𝐿 / 𝜕𝑦𝑙,       𝑖𝑓 (𝑦[𝑙] > 0)
void leaky_relu_layer::back_propagation(const float_t* in_data, const float_t* out_data,
                                  float_t* out_grad, float_t* in_grad) {
  galois::do_all(galois::iterate((size_t)0, n), [&](const auto& i) {
    in_grad[i] = out_grad[i] * (out_data[i] > float_t(0) ? float_t(1) : epsilon_);
  }, galois::chunk_size<64>(), galois::steal(), galois::loopname("leaky_relu_layer-bw"));
}
#endif

} // namespace
