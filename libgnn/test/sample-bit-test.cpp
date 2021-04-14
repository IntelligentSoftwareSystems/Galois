//! @file sample-bit-test.cpp
//! Checks to see if edge sample bit is set correctly.

#include "galois/Logging.h"
#include "galois/graphs/GNNGraph.h"

int main() {
  galois::DistMemSys G;

  size_t num_threads = galois::setActiveThreads(
      56 / galois::runtime::getSystemNetworkInterface().Num);
  GALOIS_LOG_VERBOSE("[{}] Using {} threads",
                     galois::runtime::getSystemNetworkInterface().ID,
                     num_threads);

  galois::graphs::GNNGraph graph(
      "tester", galois::graphs::GNNPartitionScheme::kOEC, true);

  // first, assert all edges are not sampled (should come with all 0s)
  for (size_t node = 0; node < graph.size(); node++) {
    for (auto ei : graph.edges(node)) {
      GALOIS_LOG_ASSERT(!graph.IsEdgeSampled(ei));
    }
    for (auto ei : graph.in_edges(node)) {
      GALOIS_LOG_ASSERT(!graph.IsInEdgeSampled(ei));
    }
  }

  // make all edges sampled; it should set the in-edges as well
  for (size_t node = 0; node < graph.size(); node++) {
    for (auto ei : graph.edges(node)) {
      graph.MakeEdgeSampled(ei);
    }
  }

  // all edges (including ins) should be sampled
  for (size_t node = 0; node < graph.size(); node++) {
    for (auto ei : graph.edges(node)) {
      GALOIS_LOG_ASSERT(graph.IsEdgeSampled(ei));
    }
    for (auto ei : graph.in_edges(node)) {
      GALOIS_LOG_ASSERT(graph.IsInEdgeSampled(ei));
    }
  }

  // clear sample bits for odd numbers
  for (size_t node = 0; node < graph.size(); node++) {
    if (node % 2 == 1) {
      for (auto ei : graph.edges(node)) {
        graph.MakeEdgeUnsampled(ei);
      }
    }
  }

  // do another check
  for (size_t node = 0; node < graph.size(); node++) {
    for (auto ei : graph.edges(node)) {
      if (node % 2 == 1) {
        GALOIS_LOG_ASSERT(!graph.IsEdgeSampled(ei));
      } else {
        GALOIS_LOG_ASSERT(graph.IsEdgeSampled(ei));
      }
    }

    // in edges for this node: if destination (i.e., source) is
    // odd, then it should not be sampled
    for (auto ei : graph.in_edges(node)) {
      if ((graph.GetInEdgeDest(ei) % 2) == 1) {
        GALOIS_LOG_ASSERT(!graph.IsInEdgeSampled(ei));
      } else {
        GALOIS_LOG_ASSERT(graph.IsInEdgeSampled(ei));
      }
    }
  }

  // print edges for a quick lookover if run manually
  for (size_t node = 0; node < graph.size(); node++) {
    for (auto ei : graph.edges(node)) {
      galois::gPrint("Out edge ", node, " ", graph.GetEdgeDest(ei), "\n");
    }
    for (auto ei : graph.in_edges(node)) {
      galois::gPrint("In edge to ", node, " from ", graph.GetInEdgeDest(ei),
                     "\n");
    }
  }

  return 0;
}
