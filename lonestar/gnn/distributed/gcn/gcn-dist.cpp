#include "GNNBench/Start.h"
#include "galois/GraphNeuralNetwork.h"

constexpr static const char* const name = "Graph Convolutional Network";

int main(int argc, char* argv[]) {
  galois::DistMemSys G;
  GNNBenchStart(argc, argv, name);
  return 0;
}
