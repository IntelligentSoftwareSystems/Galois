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
  CUDA_FREE(activation_memo_);
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

void galois::GNNLayerGPUAllocations::CopyToWeightGradients(
    const std::vector<GNNFloat>& cpu_gradients) {
  CUDA_CHECK(cudaMemcpy(layer_weight_gradients_, cpu_gradients.data(),
                        cpu_gradients.size() * sizeof(GNNFloat),
                        cudaMemcpyHostToDevice));
}

void galois::GNNLayerGPUAllocations::CopyForwardOutputToCPU(
    GNNFloat* cpu_forward_output, size_t forward_output_size) {
  CUDA_CHECK(cudaMemcpy(cpu_forward_output, forward_output_matrix_,
                        forward_output_size * sizeof(GNNFloat),
                        cudaMemcpyDeviceToHost));
}

void galois::GNNLayerGPUAllocations::CopyBackwardOutputToCPU(
    GNNFloat* cpu_backward_output, size_t backward_output_size) {
  CUDA_CHECK(cudaMemcpy(cpu_backward_output, backward_output_matrix_,
                        backward_output_size * sizeof(GNNFloat),
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
  CUDA_KERNEL_LOOP(i, 0, input_size) {
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
  CUDA_KERNEL_LOOP(i, 0, input_size) {
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

// TODO copy from gpu function as well just in case I need to check
void galois::GNNLayerGPUAllocations::PrintBackwardOutput(size_t size) {
  PrintVector<<<1, 1>>>(backward_output_matrix_, size);
}

namespace {
__global__ void InitVectorTo1Kernel(galois::GNNFloat* vector,
                                    size_t num_vector_elements) {
  CUDA_KERNEL_LOOP(idx, 0, num_vector_elements) { vector[idx] = 1.0; }
}

__global__ void ReluActivationKernel(galois::GNNFloat* forward_output_matrix,
                                     size_t num_forward_output_elements,
                                     uint8_t* activation_memo) {
  CUDA_KERNEL_LOOP(idx, 0, num_forward_output_elements) {
    if (forward_output_matrix[idx] > galois::GNNFloat{0}) {
      activation_memo[idx] = 1;
    } else {
      forward_output_matrix[idx] = 0;
    }
  }
}

__global__ void ReluActivationDerivativeKernel(
    galois::GNNFloat* gradients, galois::GNNFloat* forward_output_matrix,
    const size_t num_gradients_elements, const uint8_t* activation_memo) {
  CUDA_KERNEL_LOOP(idx, 0, num_gradients_elements) {
    if (!activation_memo[idx]) {
      gradients[idx] = 0;
    }
  }
}

__global__ void
ReconstructDropoutMatrixKernel(const galois::GNNFloat* input_to_dropout,
                               galois::GNNFloat* output_matrix,
                               char* dropout_mask, const size_t num_elements,
                               const galois::GNNFloat scale) {
  CUDA_KERNEL_LOOP(i, 0, num_elements) {
    output_matrix[i] = input_to_dropout[i] * scale;
  }

  CUDA_KERNEL_LOOP(i, 0, num_elements) {
    output_matrix[i] *= static_cast<galois::GNNFloat>(dropout_mask[i]);
  }
}

__global__ void MaskNonMastersKernel(galois::GNNFloat* input,
                                     uint32_t start_node, uint32_t end_node,
                                     uint32_t row_index) {
  // TODO(lhc) implement nested parallelism if it is worth
  CUDA_KERNEL_LOOP(non_master, start_node, end_node) {
    for (uint32_t j = 0; j < row_index; j++) {
      input[non_master * row_index + j] = 0;
    }
  }
}
} // namespace

void galois::GNNLayerGPUAllocations::InitGPUVectorTo1(GNNFloat* vector,
                                                      size_t vector_size) {
  InitVectorTo1Kernel<<<CUDA_GET_BLOCKS(vector_size), CUDA_NUM_THREADS>>>(
      vector, vector_size);
  CUDA_TEST("Failed to initialize vector to 1.");
}

void galois::GNNLayerGPUAllocations::ActivationGPU(
    size_t num_forward_output_elements) {
  if (activation_memo_ == nullptr) {
    CUDA_CHECK(cudaMalloc((void**)(&activation_memo_),
                          num_forward_output_elements * sizeof(uint8_t)));
  }
  ReluActivationKernel<<<CUDA_GET_BLOCKS(num_forward_output_elements),
                         CUDA_NUM_THREADS>>>(
      forward_output_matrix_, num_forward_output_elements, activation_memo_);
  CUDA_TEST("Activation GPU failed.");
}

void galois::GNNLayerGPUAllocations::ActivationDerivativeGPU(
    GNNFloat* gradients, size_t num_gradients_elements) {
  ReluActivationDerivativeKernel<<<CUDA_GET_BLOCKS(num_gradients_elements),
                                   CUDA_NUM_THREADS>>>(
      gradients, forward_output_matrix_, num_gradients_elements,
      activation_memo_);
  CUDA_TEST("ActivationDerivative GPU failed.");
}

void galois::GNNLayerGPUAllocations::ReconstructDropoutMatrixGPU(
    const PointerWithSize<GNNFloat> input_to_dropout,
    PointerWithSize<GNNFloat>* output_matrix, size_t num_elements,
    GNNFloat scale) {
  ReconstructDropoutMatrixKernel<<<CUDA_GET_BLOCKS(num_elements),
                                   CUDA_NUM_THREADS>>>(
      input_to_dropout.data(), output_matrix->data(), dropout_mask_,
      num_elements, scale);
}

void galois::GNNLayerGPUAllocations::MaskNonMastersGPU(
    PointerWithSize<GNNFloat>* input, size_t start_node, size_t end_node,
    size_t row_index) {
  MaskNonMastersKernel<<<CUDA_GET_BLOCKS(row_index), CUDA_NUM_THREADS>>>(
      input->data(), start_node, end_node, row_index);
}

void galois::GNNLayerGPUAllocations::CopyToCPU(
    PointerWithSize<GNNFloat>* input) {
  GNNFloat* cpu_input = (GNNFloat*)malloc(sizeof(GNNFloat) * input->size());
  cudaMemcpy(cpu_input, input->data(), sizeof(GNNFloat) * input->size(),
             cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < input->size(); i++)
    fprintf(stderr, "%lu = %f\n", i, cpu_input[i]);
}

void galois::GNNLayerGPUAllocations::CopyToCPU(GNNFloat* input, size_t size) {
  GNNFloat* cpu_input = (GNNFloat*)malloc(sizeof(GNNFloat) * size);
  cudaMemcpy(cpu_input, input, sizeof(GNNFloat) * size, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < size; i++)
    fprintf(stderr, "%lu = %f\n", i, cpu_input[i]);
}
