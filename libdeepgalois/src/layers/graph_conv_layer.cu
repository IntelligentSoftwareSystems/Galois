#include "deepgalois/layers/graph_conv_layer.h"

namespace deepgalois {

void graph_conv_layer::init() {
  gconv_malloc_device(x, y, z, dropout_, dropout_mask, in_temp, out_temp, d_W, layer::d_weight_grad);
}

void graph_conv_layer::aggregate(size_t len, CSRGraph& g, const float_t* in, float_t* out) {
  #ifdef USE_CUSPARSE
  deepgalois::update_all_csrmm(len, g, in, out, norm_, norm_factor);
  #else
  deepgalois::update_all(len, g, in, out, norm_, norm_factor);
  #endif
}

void graph_conv_layer::combine(size_t dim_x, size_t dim_y, const float_t* self, const float_t* neighbors, float_t* out) {
}

// GPU forward: compute output features
void graph_conv_layer::forward_propagation(const float_t* in_data, float_t* out_data) {
  //assert(y <= 128); // currently only support feature length <= 128
  init_const_gpu(x*z, 0.0, out_temp);
  if (dropout_ && phase_ == deepgalois::net_phase::train) {
    dropout_gpu(x * y, scale_, dropout_rate_, in_data, dropout_mask, in_temp);
    sgemm_gpu(CblasNoTrans, CblasNoTrans, x, z, y, 1.0, in_temp, d_W, 0.0, out_temp);
  } else sgemm_gpu(CblasNoTrans, CblasNoTrans, x, z, y, 1.0, in_data, d_W, 0.0, out_temp);
  graph_conv_layer::aggregate(z, context->graph_gpu, out_temp, out_data);
  if (act_) relu_gpu(x * z, out_data, out_data);
}

// GPU backward: compute input gradients (in_grad) and weight gradients (d_weight_grad)
void graph_conv_layer::back_propagation(const float_t* in_data,
                                        const float_t* out_data,
                                        float_t* out_grad, float_t* in_grad) {
  if (act_) d_relu_gpu(x * z, out_grad, out_data, out_grad);
#ifdef USE_CUSPARSE
  update_all_csrmm(z, context->graph_gpu, out_grad, out_temp, norm_, norm_factor);
#else
  update_all(z, context->graph_gpu, out_grad, out_temp, norm_, norm_factor);
#endif
  if (level_ != 0) {
    sgemm_gpu(CblasNoTrans, CblasTrans, x, y, z, 1.0, out_temp, d_W, 0.0, in_grad);
    if (dropout_) d_dropout_gpu(x * y, scale_, dropout_rate_, in_grad, dropout_mask, in_grad);
  }
  sgemm_gpu(CblasTrans, CblasNoTrans, y, z, x, 1.0, in_data, out_temp, 0.0, layer::d_weight_grad);
}

} // namespace

