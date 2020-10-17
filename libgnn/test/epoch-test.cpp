//! @file epoch-test.cpp
//! Run 50 epochs of training to see if results improve.

#include "galois/Logging.h"
#include "galois/GraphNeuralNetwork.h"

int main() {
  galois::DistMemSys G;

  size_t num_threads = galois::setActiveThreads(
      56 / galois::runtime::getSystemNetworkInterface().Num);
  // size_t num_threads = galois::setActiveThreads(1);
  GALOIS_LOG_VERBOSE("Num threads is {}", num_threads);

  // load graph
  auto test_graph = std::make_unique<galois::graphs::GNNGraph>(
      "reddit", galois::graphs::GNNPartitionScheme::kCVC, true);

  std::vector<galois::GNNLayerType> layer_types = {
      galois::GNNLayerType::kGraphConvolutional,
      galois::GNNLayerType::kGraphConvolutional};
  std::vector<size_t> layer_output_sizes = {
      16, test_graph->GetNumLabelClasses(), test_graph->GetNumLabelClasses()};
  galois::GNNLayerConfig layer_config;
  layer_config.do_dropout       = true;
  layer_config.do_activation    = false;
  layer_config.do_normalization = true;
  // XXX Activation kills accuracy compared to old code, esp. for cora
  galois::GraphNeuralNetworkConfig gnn_config(
      2, layer_types, layer_output_sizes, galois::GNNOutputLayerType::kSoftmax,
      layer_config);

  std::vector<size_t> adam_sizes = {16 * test_graph->node_feature_length(),
                                    16 * test_graph->GetNumLabelClasses()};
  auto adam = std::make_unique<galois::AdamOptimizer>(adam_sizes, 2);

  auto gnn = std::make_unique<galois::GraphNeuralNetwork>(
      std::move(test_graph), std::move(adam), std::move(gnn_config));

  //////////////////////////////////////////////////////////////////////////////

  // no verification; test should be eyeballed to make sure accuracy is
  // increasing
  galois::StatTimer main_timer("Timer_0");
  main_timer.start();
  for (size_t epoch = 0; epoch < 20; epoch++) {
    const std::vector<galois::GNNFloat>* predictions = gnn->DoInference();
    gnn->GradientPropagation();
    galois::gPrint("Epoch ", epoch, ": Accuracy is ",
                   gnn->GetGlobalAccuracy(*predictions), "\n");
  }

  // check test accuracy
  gnn->SetLayerPhases(galois::GNNPhase::kTest);
  const std::vector<galois::GNNFloat>* predictions = gnn->DoInference();
  galois::gPrint("Test accuracy is ", gnn->GetGlobalAccuracy(*predictions),
                 "\n");
  main_timer.stop();
}
