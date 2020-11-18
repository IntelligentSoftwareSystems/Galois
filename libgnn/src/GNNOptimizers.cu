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
