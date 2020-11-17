#include "galois/GNNMath.cuh"
#include "galois/Logging.h"
#include "galois/layers/SoftmaxLayer.cuh"

void galois::SoftmaxLayerGPU::ForwardPhaseGPU(galois::GNNPhase phase,
                                              size_t num_nodes,
                                              size_t feature_length,
                                              const GNNFloat* input_embeddings,
                                              GNNFloat* output) {
  char* mask_to_use = ChooseMask(phase);
  CUDA_CHECK(
      cudaMemset(output, 0, num_nodes * feature_length * sizeof(GNNFloat)));
  SoftmaxCrossEntropyForward<<<CUDA_GET_BLOCKS(num_nodes), CUDA_NUM_THREADS>>>(
      mask_to_use, num_nodes, feature_length, input_embeddings, output);
  CUDA_TEST("Softmax cross entropy forward failed");
}

// void galois::SoftmaxLayerGPU::BackwardPhaseGPU() {
//
//
//
//}

// Input: in_tensor
// Input: out_tensor
// Input: out_gradients
// Output: in_gradients
// Note: although out_gradients is an input data,
//       it is not const because it can be reused
//       to hold intermediate data inside this function,
//       to avoid allocating more memory
// void galois::SoftmaxLayerGPU::Backward(const galois::GNNFloat* in_tensor,
//                                    const galois::GNNFloat* out_tensor,
//                                    galois::GNNFloat* in_gradients,
//                                    galois::GNNFloat* out_gradients) {}
