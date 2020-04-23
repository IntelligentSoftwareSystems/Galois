#pragma once
#include "layer.h"

namespace deepgalois {
class sigmoid_loss_layer : public layer {
public:
  sigmoid_loss_layer(unsigned level, std::vector<size_t> in_dims,
                     std::vector<size_t> out_dims);
  ~sigmoid_loss_layer();
  std::string layer_type() const override {
    return std::string("sigmoid_loss");
  }
  virtual void forward_propagation(const float_t* in_data, float_t* out_data);
  virtual void back_propagation(const float_t* in_data, const float_t* out_data,
                                float_t* out_grad, float_t* in_grad);
  virtual acc_t get_prediction_loss();
};
}
