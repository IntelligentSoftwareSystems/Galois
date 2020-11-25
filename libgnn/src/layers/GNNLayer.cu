#include "galois/CUDAUtil.h"
#include "galois/GNNMath.cuh"
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

void galois::GNNLayerGPUAllocations::InitDropoutMemory(size_t dropout_size) {
  CUDA_CHECK(
      cudaMalloc((void**)(&rng_results_), dropout_size * sizeof(GNNFloat)));
  CUDA_CHECK(cudaMemset(rng_results_, 0, dropout_size * sizeof(GNNFloat)));

  CUDA_CHECK(cudaMalloc((void**)(&dropout_mask_), dropout_size * sizeof(char)));
  CUDA_CHECK(cudaMemset(dropout_mask_, 0, dropout_size * sizeof(char)));
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

namespace {

__global__ void
DoDropoutImpl(size_t input_size, const galois::GNNFloat* input_to_dropout,
              galois::GNNFloat* output, const galois::GNNFloat* rng_vector,
              char* dropout_mask, float dropout_rate, galois::GNNFloat scale) {
  CUDA_KERNEL_LOOP(i, input_size) {
    // convert the rng floats into a mask
    dropout_mask[i] = rng_vector[i] > dropout_rate ? 1 : 0;
    // use mask to keep/drop weights
    output[i] = input_to_dropout[i] * (float)dropout_mask[i] * scale;
  }
}

__global__ void DoDropoutDerivativeImpl(size_t input_size,
                                        galois::GNNFloat* input,
                                        char* dropout_mask,
                                        galois::GNNFloat scale) {
  CUDA_KERNEL_LOOP(i, input_size) {
    input[i] = input[i] * (float)dropout_mask[i] * scale;
  }
}

} // namespace

void galois::GNNLayerGPUAllocations::DoDropoutGPU(
    const PointerWithSize<GNNFloat> input_to_dropout,
    PointerWithSize<GNNFloat> output, float dropout_rate) {
  // RNG which weights to dropout
  galois::CuRANDUniformRNG(rng_results_, input_to_dropout.size());
  GNNFloat scale = 1. / (1. - dropout_rate);
  // GPU dropout kernel
  DoDropoutImpl<<<CUDA_GET_BLOCKS(input_to_dropout.size()), CUDA_NUM_THREADS>>>(
      input_to_dropout.size(), input_to_dropout.data(), output.data(),
      rng_results_, dropout_mask_, dropout_rate, scale);
  CUDA_TEST("Dropout on GPU failure");
}

void galois::GNNLayerGPUAllocations::DoDropoutDerivativeGPU(size_t input_size,
                                                            GNNFloat scale) {
  DoDropoutDerivativeImpl<<<CUDA_GET_BLOCKS(input_size), CUDA_NUM_THREADS>>>(
      input_size, backward_output_matrix_, dropout_mask_, scale);
  CUDA_TEST("Dropout derivative on GPU failure");
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
