#include "deepgalois/layers/l2_norm_layer.h"
#include "deepgalois/math_functions.hh"

namespace deepgalois {

void l2_norm_layer::forward_propagation(const float_t* in_data, float_t* out_data) {
  size_t x = input_dims[0];
  size_t y = input_dims[1];
  l2_norm_gpu(x, y, in_data, out_data);
}

void l2_norm_layer::back_propagation(const float_t* in_data, const float_t* out_data,
                                  float_t* out_grad, float_t* in_grad) {
  size_t x = input_dims[0];
  size_t y = input_dims[1];
  d_l2_norm_gpu(x, y, in_data, out_grad, in_grad);
}

} // namespace
