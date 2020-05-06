#include "deepgalois/layers/node.h"
#include <iostream>

namespace deepgalois {

void edge::alloc() {
  data_ = new float_t[num_samples_ * ft_dim_];
  grad_ = new float_t[num_samples_ * ft_dim_];
}

void edge::merge_grads(float_t* dst) {
  assert(grad_ != NULL);
  if(dst) delete[] dst;
  dst = new float_t[ft_dim_];
  std::copy(grad_, grad_ + ft_dim_, dst);
  // @todo consider adding parallelism and vectorization
  for (size_t sample = 1; sample < num_samples_; ++sample) {
    for (size_t i = 0; i < ft_dim_; i++)
      dst[i] += grad_[sample * ft_dim_ + i];
  }
}

void edge::clear_grads() {
  std::fill(grad_, grad_ + ft_dim_ * num_samples_, float_t(0));
}

} // namespace deepgalois
