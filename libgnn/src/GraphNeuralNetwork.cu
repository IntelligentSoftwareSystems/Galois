#include "galois/GraphNeuralNetwork.cuh"
#include "galois/Logging.h"

float galois::GraphNeuralNetworkGPU::GetGlobalAccuracyGPU(
    const graphs::GNNGraphGPUAllocations& gpu_graph, GNNPhase phase,
    const PointerWithSize<GNNFloat> predictions) {
  // get correct mask
  char* mask_to_use = nullptr;
  switch (phase) {
  case GNNPhase::kTrain:
    mask_to_use = gpu_graph.local_training_mask();
    break;
  case GNNPhase::kValidate:
    mask_to_use = gpu_graph.local_validation_mask();
    break;
  case GNNPhase::kTest:
    mask_to_use = gpu_graph.local_testing_mask();
    break;
  default:
    GALOIS_LOG_FATAL("Invalid phase specified");
  }

  // run accuracy check kernel on GPU

  return 0.0;
}
