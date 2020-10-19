#include "GNNBench/Start.h"

constexpr static const char* const name = "Graph Convolutional Network";

int main(int argc, char* argv[]) {
  galois::DistMemSys G;
  GNNBenchStart(argc, argv, name);

  galois::StatTimer init_timer("InitializationTime");
  init_timer.start();
  std::unique_ptr<galois::GraphNeuralNetwork> gnn =
    InitializeGraphNeuralNetwork(galois::GNNLayerType::kGraphConvolutional);
  gnn->SetLayerPhases(galois::GNNPhase::kTrain);
  init_timer.stop();

  galois::StatTimer compute_timer("Timer_0");
  compute_timer.start();

  galois::StatTimer train_timer("TrainingTime");
  train_timer.start();

  train_timer.stop();
  compute_timer.stop();

  return 0;
}
