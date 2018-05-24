// NOTE: must be before DistBenchStart.h as that relies on some cuda
// calls

#include <iostream>
#include <limits>
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"
#include "galois/DReducible.h"
#include "galois/runtime/Tracer.h"

struct NodeData {
  char blah;
};

typedef galois::graphs::DistGraph<NodeData, unsigned> Graph;
typedef galois::graphs::DistGraph_edgeCut<NodeData, unsigned> Graph_edgeCut;
typedef typename Graph::GraphNode GNode;

/******************************************************************************/
/* Main */
/******************************************************************************/

constexpr static const char* const name = "Check Transpose";
constexpr static const char* const desc = "Transpose check.";
constexpr static const char* const url = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  const auto& net = galois::runtime::getSystemNetworkInterface();
  std::vector<unsigned> dummyScale;

  masters_distribution = BALANCED_MASTERS;
  partitionScheme = OEC;

  Graph* regular = new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num,
                                     dummyScale, false);
  Graph* flipped = new Graph_edgeCut(inputFileTranspose, partFolder, net.ID, net.Num,
                                     dummyScale, false);

  galois::gPrint("Graphs loaded: beginning transpose checking of all edges.\n");

  galois::do_all(
    galois::iterate(regular->masterNodesRange()),
    [&] (auto node) {
      for (auto edge : regular->edges(node)) {
        auto edgeDst = regular->getEdgeDst(edge);
        auto edgeData = regular->getEdgeData(edge);

        auto globalNodeID = regular->getGID(edgeDst);

        // if the transpose owns the node, then 
        if (flipped->isOwned(globalNodeID)) {
          bool found = false;
          auto flippedNodeID = flipped->getLID(globalNodeID);

          // check to see if this edge exists in the other graph
          for (auto flipEdge : flipped->edges(flippedNodeID)) {
            auto flipEdgeDst = flipped->getEdgeDst(flipEdge);

            // see if it matches the original source node
            if (flipped->getGID(flipEdgeDst) == regular->getGID(node)) {
              // check edge data to see if matches
              auto flipEdgeData = flipped->getEdgeData(flipEdge);

              // onto the next edge
              if (flipEdgeData == edgeData) {
                found = true;
                break;
              } 
            }
          }

          if (!found) {
            printf("Edge %lu to %lu with same weight not found\n", 
                   regular->getGID(node), regular->getGID(edgeDst));
            GALOIS_DIE("An edge was not found");
          }
        }
      }
    });

  return 0;
}
