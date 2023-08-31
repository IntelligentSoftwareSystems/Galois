#include "GNNBench/Start.h"

constexpr static const char* const name = "Graph Convolutional Network";

int main(int argc, char* argv[]) {
  galois::DistMemSys G;
  GNNBenchStart(argc, argv, name);

  galois::StatTimer init_timer("InitializationTime");
  init_timer.start();
  std::unique_ptr<
      galois::GraphNeuralNetwork<shad::ShadNodeTy, shad::ShadEdgeTy>> gnn =
      InitializeGraphNeuralNetwork<shad::ShadNodeTy, shad::ShadEdgeTy>();
  gnn->SetLayerPhases(galois::GNNPhase::kTrain);
  init_timer.stop();

  galois::runtime::getHostBarrier().wait();

  galois::StatTimer compute_timer("Timer_0");
  compute_timer.start();
  gnn->Train(num_epochs);
  compute_timer.stop();

  return 0;
}
