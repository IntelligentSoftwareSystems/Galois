//! @file gnnconstruct-test.cpp
//! Test to make sure construction works as expected

#include "galois/Logging.h"
#include "galois/GraphNeuralNetwork.h"

int main() {
  galois::DistMemSys G;

  size_t num_threads = galois::setActiveThreads(
      56 / galois::runtime::getSystemNetworkInterface().Num);

  GALOIS_LOG_VERBOSE("[{}] Using {} threads",
                     galois::runtime::getSystemNetworkInterface().ID,
                     num_threads);
  // load test graph
  auto test_graph = std::make_unique<galois::graphs::GNNGraph<char, void>>(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, true, false);

  // 2 layer test with softmax
  std::vector<galois::GNNLayerType> layer_types = {
      galois::GNNLayerType::kGraphConvolutional,
      galois::GNNLayerType::kGraphConvolutional};
  // note this includes the output; last 2 must be same because softmax
  std::vector<size_t> layer_output_sizes = {4, 7, 7};
  galois::GraphNeuralNetworkConfig gnn_config(
      2, layer_types, layer_output_sizes, galois::GNNOutputLayerType::kSoftmax);
  std::vector<size_t> adam_sizes = {12, 28};
  auto adam = std::make_unique<galois::AdamOptimizer>(adam_sizes, 2);

  galois::GraphNeuralNetwork<char, void> gnn(
      std::move(test_graph), std::move(adam), std::move(gnn_config));

  // note this does not include output layer
  GALOIS_LOG_ASSERT(gnn.num_intermediate_layers() == 2);
  // assert layer types
  GALOIS_LOG_ASSERT(galois::GNNLayerType::kGraphConvolutional ==
                    gnn.GetIntermediateLayer(0)->layer_type());
  GALOIS_LOG_ASSERT(galois::GNNOutputLayerType::kInvalid ==
                    gnn.GetIntermediateLayer(0)->output_layer_type());
  GALOIS_LOG_ASSERT(galois::GNNLayerType::kGraphConvolutional ==
                    gnn.GetIntermediateLayer(1)->layer_type());
  GALOIS_LOG_ASSERT(galois::GNNOutputLayerType::kInvalid ==
                    gnn.GetIntermediateLayer(1)->output_layer_type());
  GALOIS_LOG_ASSERT(galois::GNNLayerType::kInvalid ==
                    gnn.GetOutputLayer()->layer_type());
  GALOIS_LOG_ASSERT(galois::GNNOutputLayerType::kSoftmax ==
                    gnn.GetOutputLayer()->output_layer_type());

  // assert dimensions are what we expect
  const galois::GNNLayerDimensions& layer0_dims =
      gnn.GetIntermediateLayer(0)->GetLayerDimensions();
  GALOIS_LOG_ASSERT(layer0_dims.input_rows == 7);
  // remember tester has features of length 3
  GALOIS_LOG_ASSERT(layer0_dims.input_columns == 3);
  GALOIS_LOG_ASSERT(layer0_dims.output_columns == 4);

  const galois::GNNLayerDimensions& layer1_dims =
      gnn.GetIntermediateLayer(1)->GetLayerDimensions();
  GALOIS_LOG_ASSERT(layer1_dims.input_rows == 7);
  GALOIS_LOG_ASSERT(layer1_dims.input_columns == 4);
  GALOIS_LOG_ASSERT(layer1_dims.output_columns == 7);

  const galois::GNNLayerDimensions& output_dims =
      gnn.GetOutputLayer()->GetLayerDimensions();
  GALOIS_LOG_ASSERT(output_dims.input_rows == 7);
  GALOIS_LOG_ASSERT(output_dims.input_columns == 7);
  GALOIS_LOG_ASSERT(output_dims.output_columns == 7);
}
