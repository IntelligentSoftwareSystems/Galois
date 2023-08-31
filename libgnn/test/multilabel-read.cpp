//! @file multilabel-read
//! Make sure multilabels read are sane

#include "galois/Logging.h"
#include "galois/graphs/GNNGraph.h"

int main() {
  galois::DistMemSys G;

  // load test graph; false at end = multilabel
  galois::graphs::GNNGraph<char, void> test_graph(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, false, false);
  const galois::GNNLabel* labels = test_graph.GetMultiClassLabel(0);

  unsigned i = 0;
  GALOIS_LOG_ASSERT(1 == labels[i * 7]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 1]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 2]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 3]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 4]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 5]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 6]);

  i = 1;
  GALOIS_LOG_ASSERT(0 == labels[i * 7]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 1]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 2]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 3]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 4]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 5]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 6]);

  i = 2;
  GALOIS_LOG_ASSERT(0 == labels[i * 7]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 1]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 2]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 3]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 4]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 5]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 6]);

  i = 3;
  GALOIS_LOG_ASSERT(0 == labels[i * 7]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 1]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 2]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 3]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 4]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 5]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 6]);

  i = 4;
  GALOIS_LOG_ASSERT(0 == labels[i * 7]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 1]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 2]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 3]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 4]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 5]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 6]);

  i = 5;
  GALOIS_LOG_ASSERT(1 == labels[i * 7]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 1]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 2]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 3]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 4]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 5]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 6]);

  i = 6;
  GALOIS_LOG_ASSERT(1 == labels[i * 7]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 1]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 2]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 3]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 4]);
  GALOIS_LOG_ASSERT(0 == labels[i * 7 + 5]);
  GALOIS_LOG_ASSERT(1 == labels[i * 7 + 6]);

  labels = test_graph.GetMultiClassLabel(0);
  GALOIS_LOG_ASSERT(1 == labels[0]);
  GALOIS_LOG_ASSERT(1 == labels[1]);
  GALOIS_LOG_ASSERT(1 == labels[2]);
  GALOIS_LOG_ASSERT(0 == labels[3]);
  GALOIS_LOG_ASSERT(0 == labels[4]);
  GALOIS_LOG_ASSERT(0 == labels[5]);
  GALOIS_LOG_ASSERT(0 == labels[6]);

  labels = test_graph.GetMultiClassLabel(1);
  GALOIS_LOG_ASSERT(0 == labels[0]);
  GALOIS_LOG_ASSERT(1 == labels[1]);
  GALOIS_LOG_ASSERT(1 == labels[2]);
  GALOIS_LOG_ASSERT(1 == labels[3]);
  GALOIS_LOG_ASSERT(0 == labels[4]);
  GALOIS_LOG_ASSERT(0 == labels[5]);
  GALOIS_LOG_ASSERT(0 == labels[6]);

  labels = test_graph.GetMultiClassLabel(2);
  GALOIS_LOG_ASSERT(0 == labels[0]);
  GALOIS_LOG_ASSERT(0 == labels[1]);
  GALOIS_LOG_ASSERT(1 == labels[2]);
  GALOIS_LOG_ASSERT(1 == labels[3]);
  GALOIS_LOG_ASSERT(1 == labels[4]);
  GALOIS_LOG_ASSERT(0 == labels[5]);
  GALOIS_LOG_ASSERT(0 == labels[6]);

  labels = test_graph.GetMultiClassLabel(3);
  GALOIS_LOG_ASSERT(0 == labels[0]);
  GALOIS_LOG_ASSERT(0 == labels[1]);
  GALOIS_LOG_ASSERT(0 == labels[2]);
  GALOIS_LOG_ASSERT(1 == labels[3]);
  GALOIS_LOG_ASSERT(1 == labels[4]);
  GALOIS_LOG_ASSERT(1 == labels[5]);
  GALOIS_LOG_ASSERT(0 == labels[6]);

  labels = test_graph.GetMultiClassLabel(4);
  GALOIS_LOG_ASSERT(0 == labels[0]);
  GALOIS_LOG_ASSERT(0 == labels[1]);
  GALOIS_LOG_ASSERT(0 == labels[2]);
  GALOIS_LOG_ASSERT(0 == labels[3]);
  GALOIS_LOG_ASSERT(1 == labels[4]);
  GALOIS_LOG_ASSERT(1 == labels[5]);
  GALOIS_LOG_ASSERT(1 == labels[6]);

  labels = test_graph.GetMultiClassLabel(5);
  GALOIS_LOG_ASSERT(1 == labels[0]);
  GALOIS_LOG_ASSERT(0 == labels[1]);
  GALOIS_LOG_ASSERT(0 == labels[2]);
  GALOIS_LOG_ASSERT(0 == labels[3]);
  GALOIS_LOG_ASSERT(0 == labels[4]);
  GALOIS_LOG_ASSERT(1 == labels[5]);
  GALOIS_LOG_ASSERT(1 == labels[6]);

  labels = test_graph.GetMultiClassLabel(6);
  GALOIS_LOG_ASSERT(1 == labels[0]);
  GALOIS_LOG_ASSERT(1 == labels[1]);
  GALOIS_LOG_ASSERT(0 == labels[2]);
  GALOIS_LOG_ASSERT(0 == labels[3]);
  GALOIS_LOG_ASSERT(0 == labels[4]);
  GALOIS_LOG_ASSERT(0 == labels[5]);
  GALOIS_LOG_ASSERT(1 == labels[6]);

  return 0;
}
