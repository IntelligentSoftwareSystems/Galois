#include "deepgalois/layers/l2_norm_layer.h"
#include "deepgalois/math_functions.hh"

namespace deepgalois {

l2_norm_layer::l2_norm_layer(unsigned level, float_t eps, float_t scale,
                             dims_t in_dims, dims_t out_dims)
    : layer(level, in_dims, out_dims), epsilon_(eps), scale_(scale) {
  assert(input_dims[0] == output_dims[0]); // num_vertices
  trainable_ = false;
  name_ = layer_type() + "_" + std::to_string(level);
}

#ifdef CPU_ONLY
void l2_norm_layer::forward_propagation(const float_t* in_data, float_t* out_data) {
  size_t x = input_dims[0];
  size_t y = input_dims[1];
  galois::do_all(galois::iterate((size_t)0, x), [&](const auto i) {
  //for (size_t i = 0; i < x; i++) {
    float_t sum = 0.0;
    size_t idx = i * y;
    for (size_t j = 0; j < y; j++) {
      sum += in_data[idx + j] * in_data[idx + j];
    }
    sum = std::max(sum, epsilon_);
    sum = sqrt(sum);
    for (size_t j = 0; j < y; j++) {
      out_data[idx + j] = in_data[idx + j] / sum  * scale_;
    }
  }, galois::loopname("l2_norm"));
}

void l2_norm_layer::back_propagation(const float_t* in_data, const float_t*,
                                  float_t* out_grad, float_t* in_grad) {
  size_t x = input_dims[0];
  size_t y = input_dims[1];
  galois::do_all(galois::iterate((size_t)0, x), [&](const auto i) {
  //for (size_t i = 0; i < x; i++) {
    float_t sum_x2 = 0.0;
    float_t coef0_axis0 = 0, coef1_axis0 = 0;
    size_t idx = i * y;
    for (size_t j = 0; j < y; j++) {
      sum_x2 += powf(in_data[idx + j], 2);
      coef0_axis0 -= in_data[idx + j] * out_grad[idx + j];
    }
    coef1_axis0 = powf(sum_x2, -1.5);
    for (size_t j = 0; j < y; j++) {
      in_grad[idx + j] = in_data[idx + j] * coef0_axis0 * coef1_axis0
                         + out_grad[idx + j] * sum_x2 * coef1_axis0;
    }
  }, galois::loopname("d_l2_norm"));
}
#endif

} // namespace
