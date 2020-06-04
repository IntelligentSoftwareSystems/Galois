/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

//! [Define LC Graph]
typedef galois::graphs::LC_Linear_Graph<unsigned int, unsigned int> Graph;
//! [Define LC Graph]
typedef Graph::GraphNode GNode;
typedef std::pair<unsigned, GNode> UpdateRequest;

static const unsigned int DIST_INFINITY =
    std::numeric_limits<unsigned int>::max();

constexpr unsigned stepShift = 14;
Graph graph;

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional,
                                      cll::desc("<input file>"), cll::Required);

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv);

  //! [ReadGraph]
  galois::graphs::readGraph(graph, filename);
  //! [ReadGraph]

  galois::do_all(galois::iterate(graph),
                 [&](GNode n) { graph.getData(n) = DIST_INFINITY; });

  //! [OrderedByIntegerMetic in SSSPsimple]
  auto reqIndexer = [](const UpdateRequest& req) {
    return (req.first >> stepShift);
  };

  using namespace galois::worklists;
  typedef PerSocketChunkLIFO<16> PSchunk;
  typedef OrderedByIntegerMetric<decltype(reqIndexer), PSchunk> OBIM;
  //! [OrderedByIntegerMetic in SSSPPullsimple]

  galois::StatTimer T;
  T.start();
  graph.getData(*graph.begin()) = 0;
  //! [for_each in SSSPPullsimple]
  std::vector<UpdateRequest> init;
  init.reserve(std::distance(graph.edge_begin(*graph.begin()),
                             graph.edge_end(*graph.begin())));
  for (auto ii : graph.edges(*graph.begin()))
    init.push_back(std::make_pair(0, graph.getEdgeDst(ii)));

  galois::for_each(
      galois::iterate(init.begin(), init.end()),
      [&](const UpdateRequest& req, auto& ctx) {
        GNode active_node = req.second;
        unsigned& data    = graph.getData(active_node);
        unsigned newValue = data;

        //![loop over neighbors to compute new value]
        for (auto ii : graph.edges(active_node)) {
          GNode dst = graph.getEdgeDst(ii);
          newValue =
              std::min(newValue, graph.getData(dst) + graph.getEdgeData(ii));
        }
        //![set new value and add neighbors to wotklist
        if (newValue < data) {
          data = newValue;
          for (auto ii : graph.edges(active_node)) {
            GNode dst = graph.getEdgeDst(ii);
            if (graph.getData(dst) > newValue)
              ctx.push(std::make_pair(newValue, dst));
          }
        }
      },
      galois::wl<OBIM>(reqIndexer), galois::loopname("sssp_run_loop"));
  //! [for_each in SSSPPullsimple]
  T.stop();
  return 0;
}
