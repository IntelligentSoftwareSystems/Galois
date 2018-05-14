// This examples shows
// 1. how to pass a range for data-driven algorithms
// 2. how to add new work items using context
// 3. how to specify schedulers
// 4. how to write an indexer for OBIM
#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"

#include <iostream>
#include <string>

using Graph = galois::graphs::LC_Linear_Graph<unsigned int, unsigned int>;
using GNode = Graph::GraphNode;
using UpdateRequest = std::pair<unsigned, GNode>;

static const unsigned int DIST_INFINITY =
  std::numeric_limits<unsigned int>::max();

unsigned int stepShift = 11;

// functor for the operator in this example
struct SSSP {
  Graph& graph;
  SSSP(Graph& _g) :graph(_g) {}

  void operator()(GNode active_node, auto& ctx) const {
    // Get the value on the node
    unsigned int data = graph.getData(active_node);

    // loop over neighbors to compute new value
    for (auto ii : graph.edges(active_node)) {                 // cautious point
      GNode dst = graph.getEdgeDst(ii);
      if (graph.getData(dst) > graph.getEdgeData(ii) + data) {
        graph.getData(dst) = graph.getEdgeData(ii) + data;
        ctx.push(dst); // add new work items
      }
    }
  }
};

// Priority Function in SSSPPushSimple
// Map user-defined priority to a bucket number in OBIM
struct UpdateRequestIndexer {
  Graph& g;
  UpdateRequestIndexer(Graph& _g) :g(_g) {}

  unsigned int operator() (const GNode N) const {
    return g.getData(N, galois::MethodFlag::UNPROTECTED) >> stepShift;
  }
};

int main(int argc, char **argv) {
  galois::SharedMemSys G;
  galois::setActiveThreads(256); // Galois will cap at hw max

  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " filename <dchunk16|obim>\n";
    return 1;
  } else {
    std::cout << "Note: This is just a very simple example and provides no useful information for performance\n";
  }

  Graph graph;
  galois::graphs::readGraph(graph, argv[1]); // argv[1] is the file name for graph

  // initialization
  galois::do_all(
      galois::iterate(graph.begin(), graph.end()),               // range
      [&graph] (GNode& N) { graph.getData(N) = DIST_INFINITY; }  // operator as lambda expression
  );
  
  galois::StatTimer T;
  T.start();

  // clear source
  graph.getData(*graph.begin()) = 0;

  // pack source to a container to have iterators for galois::iterate
  std::array<GNode,1> init = {*graph.begin()};

  using namespace galois::worklists;
  using dChunk = dChunkedLIFO<16>;
  using OBIM = OrderedByIntegerMetric<UpdateRequestIndexer,dChunk>;

  std::string schedule = argv[2];  // argv[2] is the scheduler to be used

  //*****************************************************************************************
  // for_each iterates over init to compute SSSP on graph using dChunk scheduler
  // chunk size of dChunk worklist is 16
  if ("dchunk16" == schedule) {
    galois::for_each(
        galois::iterate(init.begin(), init.end()), // range
        SSSP{graph}                                // operator as functor
        , galois::wl<dChunk>()                     // options
        , galois::loopname("sssp_dchunk16")
    );

  //*****************************************************************************************
  // for_each iterates over init to compute SSSP on graph using OBIM scheduler
  // each bin in OBIM is a dChunk worklist
  // the OBIM in this example uses UpdateRequestIndexer as its indexer
  } else if ("obim" == schedule) {
    galois::for_each(
        galois::iterate(init.begin(), init.end()),      // range
        SSSP{graph}                                     // operator as functor
        , galois::wl<OBIM>(UpdateRequestIndexer{graph}) // options
        , galois::loopname("sssp_obim")
    );

  } else {
    std::cerr << "Unknown schedule " << schedule << std::endl;
    return 1;
  }

  T.stop();
  return 0;
}
