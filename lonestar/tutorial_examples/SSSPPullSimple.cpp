/** Single source shortest paths -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * Single source shortest paths, pull based.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
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

unsigned stepShift = 11;
Graph graph;

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);

//! [Operator in SSSPPullsimple]
struct SSSP {
  void operator()(UpdateRequest& req, auto& ctx) const {
    GNode active_node = req.second;
    unsigned& data = graph.getData(active_node);
    unsigned newValue = data;

    //![loop over neighbors to compute new value] 
    for (auto ii : graph.edges(active_node)) {
      GNode dst = graph.getEdgeDst(ii);
      newValue = std::min(newValue, graph.getData(dst) + graph.getEdgeData(ii));
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
  }
};
//! [Operator in SSSPPullsimple]

struct Init {
  void operator()(GNode& n, auto& ctx) const {
    graph.getData(n) = DIST_INFINITY;
  }
};


int main(int argc, char **argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, 0,0,0);

//! [ReadGraph]
  galois::graphs::readGraph(graph, filename);
//! [ReadGraph]

  galois::for_each(galois::iterate(graph.begin(), graph.end()), Init());

  //! [OrderedByIntegerMetic in SSSPsimple]
  struct UpdateRequestIndexer: public std::unary_function<UpdateRequest, unsigned int> {
    unsigned int operator() (const UpdateRequest& val) const {
      return val.first >> stepShift;
    }
  };
  using namespace galois::worklists;
  typedef dChunkedLIFO<16> dChunk;
  typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk> OBIM;
//! [OrderedByIntegerMetic in SSSPPullsimple]

  galois::StatTimer T;
  T.start();
  graph.getData(*graph.begin()) = 0;
  //! [for_each in SSSPPullsimple]
  std::vector<UpdateRequest> init;
  init.reserve(std::distance(graph.edge_begin(*graph.begin()), graph.edge_end(*graph.begin())));
  for (auto ii : graph.edges(*graph.begin()))
    init.push_back(std::make_pair(0, graph.getEdgeDst(ii)));

  galois::for_each(galois::iterate(init.begin(), init.end()), SSSP(), galois::wl<OBIM>(), galois::loopname("sssp_run_loop"));
  //! [for_each in SSSPPullsimple]
  T.stop();
  return 0;
}
