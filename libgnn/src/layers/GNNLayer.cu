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

void galois::GNNLayerGPUAllocations::CopyForwardOutputToCPU(
    std::vector<GNNFloat>* cpu_forward_output) {
  CUDA_CHECK(cudaMemcpy(cpu_forward_output->data(), forward_output_matrix_,
                        cpu_forward_output->size() * sizeof(GNNFloat),
                        cudaMemcpyDeviceToHost));
}

void galois::GNNLayerGPUAllocations::CopyBackwardOutputToCPU(
    std::vector<GNNFloat>* cpu_backward_output) {
  CUDA_CHECK(cudaMemcpy(cpu_backward_output->data(), backward_output_matrix_,
                        cpu_backward_output->size() * sizeof(GNNFloat),
                        cudaMemcpyDeviceToHost));
}

void galois::GNNLayerGPUAllocations::CopyWeightGradientsToCPU(
    std::vector<GNNFloat>* cpu_gradients) {
  CUDA_CHECK(cudaMemcpy(cpu_gradients->data(), layer_weight_gradients_,
                        cpu_gradients->size() * sizeof(GNNFloat),
                        cudaMemcpyDeviceToHost));
}

galois::GNNFloat*
galois::GNNLayerGPUAllocations::Allocate(const std::vector<GNNFloat>& v) {
  // TODO keep track of these so that on destruction they can be freed
  // accordingly; for now I'll let them leak
  galois::GNNFloat* to_return = nullptr;
  CUDA_CHECK(
      cudaMalloc((void**)(&to_return), v.size() * sizeof(galois::GNNFloat)));
  CUDA_CHECK(cudaMemcpy(to_return, v.data(),
                        v.size() * sizeof(galois::GNNFloat),
                        cudaMemcpyHostToDevice));
  return to_return;
}

namespace {
__global__ void PrintVector(galois::GNNFloat* v, unsigned size) {
  for (unsigned i = 0; i < size; i++) {
    printf("%u %f\n", i, v[i]);
  }
}
} // namespace

// TODO copy from gpu function as well just in case I need to check
void galois::GNNLayerGPUAllocations::PrintForwardOutput(size_t size) {
  PrintVector<<<1, 1>>>(forward_output_matrix_, size);
}
