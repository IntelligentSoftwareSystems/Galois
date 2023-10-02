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

  // note multi level reading tested in another test
  GALOIS_LOG_VERBOSE("reddit with single label, oec");
  galois::graphs::GNNGraph<char, void>(
      "cora", galois::graphs::GNNPartitionScheme::kOEC, true, false);
  GALOIS_LOG_VERBOSE("reddit with single label, cvc");
  galois::graphs::GNNGraph<char, void>(
      "cora", galois::graphs::GNNPartitionScheme::kCVC, true, false);

  // below for when I want to check the remapper
  // galois::graphs::GNNGraph remapper("ogbn-papers100M",
  // galois::graphs::GNNPartitionScheme::kOEC, true);
  // remapper.ContiguousRemap("ogbn-papers100M-remap");
  // galois::graphs::GNNGraph remapper("ogbn-papers100M-remap",
  // galois::graphs::GNNPartitionScheme::kOEC, true);

  // galois::graphs::GNNGraph remapper("yelp",
  // galois::graphs::GNNPartitionScheme::kOEC, true);
  // remapper.ContiguousRemap("yelp-remap");

  return 0;
}
