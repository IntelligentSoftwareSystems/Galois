//! @file gnngraph-test.cpp
//! Test loads a few graphs. Better if you run with multiple hosts.
//! Doesn't really do much besides that.

#include "galois/Logging.h"
#include "galois/graphs/GNNGraph.h"

int main() {
  galois::DistMemSys G;

  size_t num_threads = galois::setActiveThreads(
      56 / galois::runtime::getSystemNetworkInterface().Num);
  GALOIS_LOG_VERBOSE("[{}] Using {} threads",
                     galois::runtime::getSystemNetworkInterface().ID,
                     num_threads);

  GALOIS_LOG_VERBOSE("reddit with multilabel, oec");
  galois::graphs::GNNGraph("reddit", galois::graphs::GNNPartitionScheme::kOEC,
                           false);
  GALOIS_LOG_VERBOSE("reddit with single label, oec");
  galois::graphs::GNNGraph("reddit", galois::graphs::GNNPartitionScheme::kOEC,
                           true);
  GALOIS_LOG_VERBOSE("reddit with multilabel, cvc");
  galois::graphs::GNNGraph("reddit", galois::graphs::GNNPartitionScheme::kCVC,
                           false);
  GALOIS_LOG_VERBOSE("reddit with single label, cvc");
  galois::graphs::GNNGraph("reddit", galois::graphs::GNNPartitionScheme::kCVC,
                           true);

  // TODO fix citeseer and goec
  // galois::graphs::GNNGraph("citeseer",
  // galois::graphs::GNNPartitionScheme::kOEC, false);
  return 0;
}
