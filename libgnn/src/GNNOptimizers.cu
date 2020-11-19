#include "galois/GNNOptimizers.cuh"
#include "galois/CUDAUtil.h"

galois::AdamOptimizerGPU::AdamOptimizerGPU(
    const std::vector<size_t>& trainable_layer_sizes, size_t num_trainable) {
  num_layers_ = num_trainable;
  first_moments_.resize(num_layers_);
  second_moments_.resize(num_layers_);

  for (size_t layer = 0; layer < num_layers_; layer++) {
    // initialize the moment vector memory then zero it all out
    CUDA_CHECK(cudaMalloc((void**)(&(first_moments_[layer])),
                          trainable_layer_sizes[layer] * sizeof(GNNFloat)));
    CUDA_CHECK(cudaMalloc((void**)(&(second_moments_[layer])),
                          trainable_layer_sizes[layer] * sizeof(GNNFloat)));
    CUDA_CHECK(cudaMemset(first_moments_[layer], 0,
                          trainable_layer_sizes[layer] * sizeof(GNNFloat)));
    CUDA_CHECK(cudaMemset(second_moments_[layer], 0,
                          trainable_layer_sizes[layer] * sizeof(GNNFloat)));
  }
}

galois::AdamOptimizerGPU::~AdamOptimizerGPU() {
  // loop through and free first/second moments
  for (size_t layer = 0; layer < num_layers_; layer++) {
    CUDA_FREE(first_moments_[layer]);
    CUDA_FREE(second_moments_[layer]);
  }
}
void galois::AdamOptimizerGPU::CopyToVector(std::vector<GNNFloat>& to,
                                            PointerWithSize<GNNFloat> from) {
  CUDA_CHECK(cudaMemcpy(to.data(), from.data(), to.size() * sizeof(GNNFloat),
                        cudaMemcpyDeviceToHost));
}

namespace {

__global__ void DoAdamUpdate(const galois::GNNFloat* derivatives,
                             galois::GNNFloat* matrix_to_update,
                             size_t matrix_size, galois::GNNFloat* first_moment,
                             galois::GNNFloat* second_moment,
                             galois::GNNFloat alpha, galois::GNNFloat beta1,
                             galois::GNNFloat beta2, galois::GNNFloat epsilon,
                             galois::GNNFloat beta1t, galois::GNNFloat beta2t) {
  CUDA_KERNEL_LOOP(i, matrix_size) {
    first_moment[i]  = beta1 * first_moment[i] + (1.0 - beta1) * derivatives[i];
    second_moment[i] = beta2 * second_moment[i] +
                       (1.0 - beta2) * (derivatives[i] * derivatives[i]);
    // bias corrected moments using beta power
    galois::GNNFloat bias_correct_first  = first_moment[i] / (1.0 - beta1t);
    galois::GNNFloat bias_correct_second = second_moment[i] / (1.0 - beta2t);
    // weight update using bias corrected moments
    matrix_to_update[i] -=
        alpha * bias_correct_first / sqrtf(bias_correct_second + epsilon);
  }
}

} // namespace

void galois::AdamOptimizerGPU::AdamUpdate(
    const GNNFloat* derivatives, GNNFloat* matrix_to_update, size_t matrix_size,
    GNNFloat* first_moment, GNNFloat* second_moment, GNNFloat alpha,
    GNNFloat beta1, GNNFloat beta2, GNNFloat epsilon, GNNFloat beta1t,
    GNNFloat beta2t) {
  DoAdamUpdate<<<CUDA_GET_BLOCKS(matrix_size), CUDA_NUM_THREADS>>>(
      derivatives, matrix_to_update, matrix_size, first_moment, second_moment,
      alpha, beta1, beta2, epsilon, beta1t, beta2t);
}
