#include <iostream>
#include <limits>
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"
#include "galois/DReducible.h"
#include "galois/runtime/Tracer.h"

struct NodeData {
  uint32_t blah;
};

typedef galois::graphs::DistGraph<NodeData, unsigned> Graph;
typedef galois::graphs::DistGraph_edgeCut<NodeData, unsigned> Graph_edgeCut;

/******************************************************************************/
/* Main */
/******************************************************************************/

constexpr static const char* const name = "Check Weight";
constexpr static const char* const desc = "Weight check.";
constexpr static const char* const url = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  const auto& net = galois::runtime::getSystemNetworkInterface();
  std::vector<unsigned> dummyScale;

  partitionScheme = OEC;

  Graph* g = new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num,
                                     dummyScale, false);

  // loop over all nodes + edges and make sure weights are between 1 and 100
  galois::do_all(
    galois::iterate(g->masterNodesRange()),
    [&] (auto node) {
      for (auto edge : g->edges(node)) {
        unsigned edgeData = g->getEdgeData(edge);
        GALOIS_ASSERT(1 <= edgeData && edgeData <= 100, ": ", edgeData, 
                      " not between 1 and 100");
      }
    }
  );

  return 0;
}
