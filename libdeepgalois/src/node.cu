#include "deepgalois/layers/node.h"
#include "deepgalois/cutils.h"
#include "deepgalois/math_functions.hh"

namespace deepgalois {

void edge::alloc() {
  CUDA_CHECK(
      cudaMalloc((void**)&data_, num_samples_ * ft_dim_ * sizeof(float_t)));
  CUDA_CHECK(
      cudaMalloc((void**)&grad_, num_samples_ * ft_dim_ * sizeof(float_t)));
}

void edge::merge_grads(float_t* dst) {
  CUDA_CHECK(cudaMemcpy(&dst, grad_, ft_dim_ * sizeof(float_t),
                        cudaMemcpyDeviceToHost));
}

void edge::clear_grads() {
  // CUDA_CHECK(cudaMemset(grad_, 0, num_samples_ * ft_dim_ * sizeof(float_t)));
  init_const_gpu(num_samples_ * ft_dim_, 0.0, grad_);
}

} // namespace deepgalois
