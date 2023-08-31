//! @file f1-test
//! Tests f1 micro accuracy for multiclass labels

#include "galois/Logging.h"
#include "galois/graphs/GNNGraph.h"

int main() {
  galois::DistMemSys G;

  // load test graph; false at end = multilabel
  galois::graphs::GNNGraph<char, void> test_graph(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, false, false);

  // perfect precision and recall
  std::vector<galois::GNNFloat> prediction = {
      1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1,
      1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1};
  GALOIS_LOG_ASSERT(1.0 == test_graph.GetGlobalAccuracy(
                               prediction, galois::GNNPhase::kTrain));
  GALOIS_LOG_ASSERT(1.0 == test_graph.GetGlobalAccuracy(
                               prediction, galois::GNNPhase::kValidate));
  GALOIS_LOG_ASSERT(
      1.0 == test_graph.GetGlobalAccuracy(prediction, galois::GNNPhase::kTest));

  // perfect recall, but training precision is bad
  std::vector<galois::GNNFloat> prediction2 = {
      1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1};

  // just print here and check with eyes: checking float equivalance is a pain
  // both prints should be .6666666
  GALOIS_LOG_DEBUG(
      "{} {}",
      test_graph.GetGlobalAccuracy(prediction2, galois::GNNPhase::kTrain),
      (2 * (15.0 / 30.0)) / ((15.0 / 30.0) + 1));
  GALOIS_LOG_ASSERT(1.0 == test_graph.GetGlobalAccuracy(
                               prediction2, galois::GNNPhase::kValidate));
  GALOIS_LOG_ASSERT(1.0 == test_graph.GetGlobalAccuracy(
                               prediction2, galois::GNNPhase::kTest));

  // no predictions made
  std::vector<galois::GNNFloat> prediction3(49, 0);
  GALOIS_LOG_ASSERT(0.0 == test_graph.GetGlobalAccuracy(
                               prediction3, galois::GNNPhase::kTrain));
  GALOIS_LOG_ASSERT(0.0 == test_graph.GetGlobalAccuracy(
                               prediction3, galois::GNNPhase::kValidate));
  GALOIS_LOG_ASSERT(0.0 == test_graph.GetGlobalAccuracy(
                               prediction3, galois::GNNPhase::kTest));

  return 0;
}
