/** BFS -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Compute BFS push on distributed Galois.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 * @author Roshan Dathathri <roshan@cs.utexas.edu>
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

// NOTE: must be before DistBenchStart.h as that relies on some cuda
// calls

#include <iostream>
#include <limits>
#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"
#include "galois/DistAccumulator.h"
#include "galois/runtime/Tracer.h"

struct NodeData {
  uint32_t blah;
};

typedef hGraph<NodeData, unsigned> Graph;
typedef hGraph_edgeCut<NodeData, unsigned> Graph_edgeCut;
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

  Graph* regular = new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num,
                                     dummyScale, false);
  Graph* flipped = new Graph_edgeCut(inputFileTranspose, partFolder, net.ID, net.Num,
                                     dummyScale, false);

  galois::do_all(
    galois::iterate(regular->masterNodesRange()),
    [&] (auto node) {
      for (auto edge : regular->edges(node)) {
        auto edgeDst = regular->getEdgeDst(edge);
        auto edgeData = regular->getEdgeData(edge);

        auto globalNodeID = regular->getGID(edgeDst);

        if (flipped->isOwned(globalNodeID)) {
          //printf("%lu owned by transpose\n", flipped->getGID(edgeDst));
          bool found = false;
          auto flippedNodeID = flipped->getLID(globalNodeID);

          // check to see if this edge exists in the other graph
          for (auto flipEdge : flipped->edges(flippedNodeID)) {
            auto flipEdgeDst = flipped->getEdgeDst(flipEdge);

            // hit this node
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
            printf("%lu to %lu not found\n", regular->getGID(node), 
                                             regular->getGID(edgeDst));
            GALOIS_DIE("edge not found");
          }
        }
      }
    },
    galois::no_stats()
  );

  return 0;
}
