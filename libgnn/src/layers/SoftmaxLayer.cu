#include <cassert>
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

void galois::SoftmaxLayerGPU::BackwardPhaseGPU(galois::GNNPhase phase,
                                               size_t num_nodes,
                                               size_t feature_length,
                                               const GNNFloat* predictions,
                                               GNNFloat* output_gradient) {
  assert(feature_length <= MAX_NUM_CLASSES);
  char* mask_to_use = ChooseMask(phase);
  CUDA_CHECK(cudaMemset(output_gradient, 0,
                        num_nodes * feature_length * sizeof(GNNFloat)));
  // TODO check the launch parameters; this is taken directly from the original
  // code
  SoftmaxCrossEntropyBackward<<<(num_nodes - 1) / WARPS_PER_BLOCK + 1,
                                BLOCK_SIZE>>>(mask_to_use, num_nodes,
                                              feature_length, predictions,
                                              local_labels_, output_gradient);
  CUDA_TEST("Softmax cross entropy backward failed");
}
