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
  auto test_graph = std::make_unique<galois::graphs::GNNGraph<char, void>>(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, true, false);

  std::vector<galois::GNNLayerType> layer_types = {
      galois::GNNLayerType::kGraphConvolutional};
  std::vector<size_t> layer_output_sizes = {7, 7};
  galois::GraphNeuralNetworkConfig gnn_config(
      1, layer_types, layer_output_sizes, galois::GNNOutputLayerType::kSoftmax,
      galois::GNNLayerConfig());

  std::vector<size_t> adam_sizes = {21};
  auto adam = std::make_unique<galois::AdamOptimizer>(adam_sizes, 1);

  auto gnn = std::make_unique<galois::GraphNeuralNetwork<char, void>>(
      std::move(test_graph), std::move(adam), std::move(gnn_config));
  // for constancy set everything to 1
  gnn->SetAllLayerWeightsTo1();

  //////////////////////////////////////////////////////////////////////////////

  galois::PointerWithSize<galois::GNNFloat> distributions = gnn->DoInference();

  float pred_accuracy = gnn->GetGlobalAccuracy(distributions);
  GALOIS_LOG_VERBOSE("{}", pred_accuracy);
  GALOIS_LOG_ASSERT(static_cast<int>(pred_accuracy * 1000) == 333);

  // validation mode
  gnn->SetLayerPhases(galois::GNNPhase::kValidate);
  galois::PointerWithSize<galois::GNNFloat> dist2 = gnn->DoInference();
  pred_accuracy = gnn->GetGlobalAccuracy(dist2);
  GALOIS_LOG_ASSERT(pred_accuracy == static_cast<float>(0));

  // test mode
  gnn->SetLayerPhases(galois::GNNPhase::kTest);
  galois::PointerWithSize<galois::GNNFloat> dist3 = gnn->DoInference();
  pred_accuracy = gnn->GetGlobalAccuracy(dist3);
  GALOIS_LOG_ASSERT(pred_accuracy == static_cast<float>(0));

  // manufactured predictions to make sure it predicts things correctly based
  // on mode
  // prediction is correct if diagonal of the 7x7 matrix has largest value
  std::vector<galois::GNNFloat> mpred = {
      1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};

  gnn->SetLayerPhases(galois::GNNPhase::kTrain);
  pred_accuracy = gnn->GetGlobalAccuracy(mpred);
  GALOIS_LOG_VERBOSE("{}", pred_accuracy);
  GALOIS_LOG_ASSERT(static_cast<int>(pred_accuracy * 1000) == 666);

  gnn->SetLayerPhases(galois::GNNPhase::kValidate);
  pred_accuracy = gnn->GetGlobalAccuracy(mpred);
  GALOIS_LOG_ASSERT(pred_accuracy == static_cast<float>(0.5));

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
