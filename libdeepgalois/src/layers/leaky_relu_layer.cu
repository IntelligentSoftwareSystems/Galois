#include "deepgalois/layers/leaky_relu_layer.h"
#include "deepgalois/math_functions.hh"

namespace deepgalois {

// 𝑦[𝑙] = 𝑦[𝑙−1] > 0 ? 𝑦[𝑙−1]) : 𝑦[𝑙−1] * ε 
void leaky_relu_layer::forward_propagation(const float_t* in_data, float_t* out_data) {
  leaky_relu_gpu(n, epsilon_, in_data, out_data);
}

// 𝜕𝐿 / 𝜕𝑦[𝑙−1] = 𝜕𝐿 / 𝜕𝑦𝑙 * ε,   𝑖𝑓 (𝑦[𝑙] ≤ 0)
//              = 𝜕𝐿 / 𝜕𝑦𝑙,       𝑖𝑓 (𝑦[𝑙] > 0)
void leaky_relu_layer::back_propagation(const float_t* in_data, const float_t* out_data,
                                  float_t* out_grad, float_t* in_grad) {
  d_leaky_relu_gpu(n, epsilon_, out_grad, in_data, in_grad);
}

} // namespace
