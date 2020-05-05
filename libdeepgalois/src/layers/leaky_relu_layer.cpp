#include "deepgalois/layers/leaky_relu_layer.h"
#include "deepgalois/math_functions.hh"

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
  math::leaky_relu_cpu(n, epsilon_, in_data, out_data);
}

// 𝜕𝐿 / 𝜕𝑦[𝑙−1] = 𝜕𝐿 / 𝜕𝑦𝑙 * ε,   𝑖𝑓 (𝑦[𝑙] ≤ 0)
//              = 𝜕𝐿 / 𝜕𝑦𝑙,       𝑖𝑓 (𝑦[𝑙] > 0)
void leaky_relu_layer::back_propagation(const float_t*, const float_t* out_data,
                                  float_t* out_grad, float_t* in_grad) {
  math::d_leaky_relu_cpu(n, epsilon_, out_grad, out_data, in_grad);
}
#endif

} // namespace
