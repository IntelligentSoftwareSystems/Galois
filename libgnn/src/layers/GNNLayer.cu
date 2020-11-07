#include "galois/CUDAUtil.h"
#include "galois/layers/GNNLayer.cuh"

galois::GNNLayerGPUAllocations::~GNNLayerGPUAllocations() {
  GALOIS_LOG_VERBOSE("Freeing GPU layer allocations");
  CUDA_FREE(num_weights_);
  CUDA_FREE(forward_output_matrix_);
  CUDA_FREE(backward_output_matrix_);
  CUDA_FREE(layer_weights_);
  CUDA_FREE(layer_weight_gradients_);
}

void galois::GNNLayerGPUAllocations::InitInOutMemory(size_t forward_size,
                                                     size_t backward_size) {
  CUDA_CHECK(cudaMalloc((void**)(&forward_output_matrix_),
                        forward_size * sizeof(GNNFloat)));
  CUDA_CHECK(cudaMalloc((void**)(&backward_output_matrix_),
                        backward_size * sizeof(GNNFloat)));
}

void galois::GNNLayerGPUAllocations::InitWeightMemory(size_t num_weights) {
  // num weights
  CUDA_CHECK(cudaMalloc((void**)(&num_weights_), sizeof(size_t)));
  CUDA_CHECK(cudaMemcpy(num_weights_, &num_weights, sizeof(size_t),
                        cudaMemcpyHostToDevice));
  // memory
  CUDA_CHECK(
      cudaMalloc((void**)(&layer_weights_), num_weights * sizeof(GNNFloat)));
  CUDA_CHECK(cudaMalloc((void**)(&layer_weight_gradients_),
                        num_weights * sizeof(GNNFloat)));
}

void galois::GNNLayerGPUAllocations::CopyToWeights(
    const std::vector<GNNFloat>& cpu_layer_weights) {
  CUDA_CHECK(cudaMemcpy(layer_weights_, cpu_layer_weights.data(),
                        cpu_layer_weights.size() * sizeof(GNNFloat),
                        cudaMemcpyHostToDevice));
}

// TODO copy from gpu function as well just in case I need to check
