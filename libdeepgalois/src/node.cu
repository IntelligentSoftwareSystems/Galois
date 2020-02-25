#include "node.h"
#include "cutils.h"

void edge::alloc_gpu() {
  CUDA_CHECK(
      cudaMalloc((void**)&data_, num_samples_ * ft_dim_ * sizeof(float_t)));
  CUDA_CHECK(
      cudaMalloc((void**)&grad_, num_samples_ * ft_dim_ * sizeof(float_t)));
}

void edge::merge_grads_gpu(float_t* dst) {
  CUDA_CHECK(cudaMemcpy(&dst, grad_, ft_dim_ * sizeof(float_t),
                        cudaMemcpyDeviceToHost));
}

void edge::clear_grads_gpu() {
  CUDA_CHECK(cudaMemset(grad_, 0, ft_dim_ * num_samples_ * sizeof(float_t)));
}
