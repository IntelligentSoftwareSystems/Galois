//! @file accuracy-test.cpp
//! Similar to softmax test except that accuracy is checked + it constructs
//! a full network object.

#include "galois/Logging.h"
#include "galois/GraphNeuralNetwork.h"

int main() {
  galois::DistMemSys G;

  size_t num_threads = galois::setActiveThreads(
      56 / galois::runtime::getSystemNetworkInterface().Num);
  GALOIS_LOG_VERBOSE("Num threads is {}", num_threads);

  // load test graph
  auto test_graph = std::make_unique<galois::graphs::GNNGraph>(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, true);

  std::vector<galois::GNNLayerType> layer_types = {
      galois::GNNLayerType::kGraphConvolutional};
  std::vector<size_t> layer_output_sizes = {7, 7};
  galois::GraphNeuralNetworkConfig gnn_config(
      1, layer_types, layer_output_sizes, galois::GNNOutputLayerType::kSoftmax,
      galois::GNNConfig());

  std::vector<size_t> adam_sizes = {21};
  auto adam = std::make_unique<galois::AdamOptimizer>(adam_sizes, 1);

  auto gnn = std::make_unique<galois::GraphNeuralNetwork>(
      std::move(test_graph), std::move(adam), std::move(gnn_config));
  // for constancy set everything to 1
  gnn->SetAllLayerWeightsTo1();

  //////////////////////////////////////////////////////////////////////////////

  const std::vector<galois::GNNFloat>* distributions = gnn->DoInference();
  // accuracy will be 0.2: everything chooses the first 1 as the entire row
  // is the same
  float pred_accuracy = gnn->GetGlobalAccuracy(*distributions);
  GALOIS_LOG_VERBOSE("{}", pred_accuracy);
  GALOIS_LOG_ASSERT(pred_accuracy == static_cast<float>(0.2));

  // validation mode
  gnn->SetLayerPhases(galois::GNNPhase::kValidate);
  const std::vector<galois::GNNFloat>* dist2 = gnn->DoInference();
  pred_accuracy                              = gnn->GetGlobalAccuracy(*dist2);
  GALOIS_LOG_ASSERT(pred_accuracy == static_cast<float>(0.0));

  // test mode
  gnn->SetLayerPhases(galois::GNNPhase::kTest);
  const std::vector<galois::GNNFloat>* dist3 = gnn->DoInference();
  pred_accuracy                              = gnn->GetGlobalAccuracy(*dist3);
  GALOIS_LOG_ASSERT(pred_accuracy == static_cast<float>(0.0));

  // manufactured predictions to make sure it predicts things correctly based
  // on mode
  // prediction is correct if diagonal of the 7x7 matrix has largest value
  std::vector<galois::GNNFloat> mpred = {
      1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};

  gnn->SetLayerPhases(galois::GNNPhase::kTrain);
  pred_accuracy = gnn->GetGlobalAccuracy(mpred);
  GALOIS_LOG_VERBOSE("{}", pred_accuracy);
  GALOIS_LOG_ASSERT(pred_accuracy == static_cast<float>(0.8));

  gnn->SetLayerPhases(galois::GNNPhase::kValidate);
  pred_accuracy = gnn->GetGlobalAccuracy(mpred);
  GALOIS_LOG_ASSERT(pred_accuracy == static_cast<float>(0.0));

  gnn->SetLayerPhases(galois::GNNPhase::kTest);
  pred_accuracy = gnn->GetGlobalAccuracy(mpred);
  GALOIS_LOG_ASSERT(pred_accuracy == static_cast<float>(1.0));

  std::vector<galois::GNNFloat> mpred2 = {
      0.5, 0, 0, 0, 0, 0, 0, 0,   0.3, 0, 0, 0, 0, 0, 0.1, 0, 1,
      0,   0, 0, 0, 0, 0, 0, 0.3, 0,   0, 0, 1, 0, 0, 0,   2, 0,
      0,   0, 0, 0, 0, 0, 4, 0,   0,   0, 0, 0, 0, 0, 0.1};
  pred_accuracy = gnn->GetGlobalAccuracy(mpred2);
  GALOIS_LOG_ASSERT(pred_accuracy == static_cast<float>(1.0));

  gnn->SetLayerPhases(galois::GNNPhase::kValidate);
  pred_accuracy = gnn->GetGlobalAccuracy(mpred2);
  GALOIS_LOG_ASSERT(pred_accuracy == static_cast<float>(1.0));

  gnn->SetLayerPhases(galois::GNNPhase::kTest);
  pred_accuracy = gnn->GetGlobalAccuracy(mpred2);
  GALOIS_LOG_ASSERT(pred_accuracy == static_cast<float>(1.0));
}
