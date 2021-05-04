#include <cassert>
#include "galois/GNNMath.cuh"
#include "galois/Logging.h"
#include "galois/layers/SoftmaxLayer.cuh"

void galois::SoftmaxLayerGPU::CopyToCPU(GNNFloat* input, size_t size) {
  GNNFloat* cpu_input = (GNNFloat*)malloc(sizeof(GNNFloat) * size);
  cudaMemcpy(cpu_input, input, sizeof(GNNFloat) * size, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < size; i++)
    fprintf(stderr, "%lu = %f\n", i, cpu_input[i]);
}

void galois::SoftmaxLayerGPU::ForwardPhaseGPU(galois::GNNPhase phase,
                                              size_t num_nodes,
                                              size_t feature_length,
                                              const GNNFloat* input_embeddings,
                                              GNNFloat* output) {
  char* mask_to_use = ChooseMask(phase);
  SoftmaxCrossEntropyForward<<<CUDA_GET_BLOCKS(num_nodes), CUDA_NUM_THREADS>>>(
      mask_to_use, num_nodes, feature_length, input_embeddings, output);
  CUDA_TEST("Softmax cross entropy forward failed");
}

__global__ void SoftmaxBackward(char* mask, size_t num_nodes,
                                size_t feature_length,
                                const galois::GNNFloat* predictions,
                                const galois::GNNLabel* ground_truth,
                                galois::GNNFloat* output_gradient) {
  const unsigned global_thread_id =
      BLOCK_SIZE * blockIdx.x + threadIdx.x; // global thread index
  const unsigned warp_thread_lane =
      threadIdx.x & (WARP_SIZE - 1); // thread index within the warp
  const unsigned warp_id = global_thread_id / WARP_SIZE; // global warp index
  const unsigned num_warps =
      (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps

  // a warp works on a single node at once
  for (unsigned wid = warp_id; wid < num_nodes; wid += num_warps) {
    // operate only if masked
    if (mask[wid] == 1) {
      unsigned base_index = wid * feature_length;
      // TODO can refactor below to device functions
      // cross entropy derivative
      // each thread of warp takes different feature
      for (unsigned feat_index = warp_thread_lane; feat_index < feature_length;
           feat_index += WARP_SIZE) {
        if (feat_index < feature_length) {
          if (feat_index == (unsigned)ground_truth[wid]) {
            output_gradient[base_index + feat_index] =
                predictions[base_index + feat_index] - 1;
          } else {
            output_gradient[base_index + feat_index] =
                predictions[base_index + feat_index];
          }
        }
      }
      __syncthreads();
    }
  }
}

void galois::SoftmaxLayerGPU::BackwardPhaseGPU(galois::GNNPhase phase,
                                               size_t num_nodes,
                                               size_t feature_length,
                                               const GNNFloat* predictions,
                                               GNNFloat* output_gradient) {
  assert(feature_length <= MAX_NUM_CLASSES);
  // num_nodes should be greater than 0 to avoid negative number of thread
  if (num_nodes == 0) {
    return;
  }

  char* mask_to_use = ChooseMask(phase);
  SoftmaxBackward<<<(num_nodes - 1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(
      mask_to_use, num_nodes, feature_length, predictions, local_labels_,
      output_gradient);

  CUDA_TEST("Softmax cross entropy backward failed");
}
