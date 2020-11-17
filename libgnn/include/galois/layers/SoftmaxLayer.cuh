#ifndef GALOIS_SOFTMAX_GPU
#define GALOIS_SOFTMAX_GPU
#include "galois/graphs/GNNGraph.cuh"
namespace galois {

//! Contains implementation for the forward/backward pass of the softmax layer
//! on GPUs.
class SoftmaxLayerGPU {
public:
  //! Initialize by saving pointers to already initialized GPU memory
  SoftmaxLayerGPU(const galois::graphs::GNNGraphGPUAllocations& gpu_graph)
      : train_mask_(gpu_graph.local_training_mask()),
        val_mask_(gpu_graph.local_validation_mask()),
        test_mask_(gpu_graph.local_testing_mask()),
        local_labels_(gpu_graph.ground_truth()) {}
  void ForwardPhaseGPU(galois::GNNPhase phase, size_t num_nodes,
                       size_t feature_length, const GNNFloat* input_embeddings,
                       GNNFloat* output);
  void BackwardPhaseGPU(GNNFloat* output);

private:
  char* train_mask_;
  char* val_mask_;
  char* test_mask_;
  GNNFloat* local_labels_;

  //! Helper function that returns the correct mask based on phase it is passed
  char* ChooseMask(galois::GNNPhase phase) {
    switch (phase) {
    case GNNPhase::kTrain:
      return train_mask_;
    case GNNPhase::kValidate:
      return val_mask_;
    case GNNPhase::kTest:
      return test_mask_;
    default:
      GALOIS_LOG_FATAL("Invalid phase specified");
      return nullptr;
    }
  }
};

} // namespace galois
#endif
