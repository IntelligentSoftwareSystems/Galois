/** Single source shortest paths -*- C++ -*-
 * @example SSSP.cpp
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
 * Single source shortest paths.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Statistic.h"
#include "Galois/Timer.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>

namespace cll = llvm::cl;

static const char* name = "Single Source Shortest Path";
static const char* desc =
    "Computes the shortest path from a source node to all nodes in a directed "
    "graph using a modified chaotic iteration algorithm";
static const char* url = "single_source_shortest_path";

static cll::opt<std::string> filename(cll::Positional, 
                                      cll::desc("<input graph>"), 
                                      cll::Required);

static cll::opt<unsigned int> startNode("startNode",
                                        cll::desc("Node to start search from"),
                                        cll::init(0));
static cll::opt<unsigned int> reportNode("reportNode", 
                                         cll::desc("Node to report distance to"),
                                         cll::init(1));
static cll::opt<int> stepShift("delta",
                               cll::desc("Shift value for the deltastep"),
                               cll::init(10));

typedef Galois::Graph::LC_InlineEdge_Graph<std::atomic<unsigned int>, uint32_t>::with_no_lockable<true>::type::with_numa_alloc<true>::type Graph;
typedef Graph::GraphNode GNode;

static const bool trackWork = false;
static Galois::Statistic* BadWork;
static Galois::Statistic* WLEmptyWork;

#include "Lonestar/BFS_SSSP.h"


struct SSSP {
  Graph& graph;
  SSSP(Graph& g) : graph(g) {}
  void operator()(UpdateRequest& req,
                  Galois::UserContext<UpdateRequest>& ctx) {
    const Galois::MethodFlag flag = Galois::MethodFlag::UNPROTECTED;
    Dist sdist = graph.getData(req.n, flag);
    
    if (req.w != sdist) {
      if (trackWork)
        *WLEmptyWork += 1;
      return;
    }
    
    for (auto ii : graph.edges(req.n, flag)) {
      GNode dst = graph.getEdgeDst(ii);
      Dist d    = graph.getEdgeData(ii, flag);
      auto& ddist  = graph.getData(dst, flag);
      Dist newDist = sdist + d;
      Dist oldDist = ddist;
      while( oldDist > newDist) {
        if (ddist.compare_exchange_weak(oldDist, newDist, std::memory_order_relaxed)) {
          if (trackWork && oldDist != DIST_INFINITY)
            *BadWork += 1;
          ctx.push(UpdateRequest(dst, newDist));
          break;
        }
      }
    }
  }
};

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  if (trackWork) {
    BadWork     = new Galois::Statistic("BadWork");
    WLEmptyWork = new Galois::Statistic("EmptyWork");
  }

  Galois::StatTimer T("OverheadTime");
  T.start();
  
  Graph graph;
  GNode source, report;

  Galois::Graph::readGraph(graph, filename); 
  std::cout << "Read " << graph.size() << " nodes\n";

  if (startNode >= graph.size() || reportNode >= graph.size()) {
    std::cerr << "failed to set report: " << reportNode
              << " or failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }

  auto it = graph.begin();
  std::advance(it, startNode);
  source = *it;
  it = graph.begin();
  std::advance(it, reportNode);
  report = *it;

  size_t approxNodeData = graph.size() * 64;
  // size_t approxEdgeData = graph.sizeEdges() * sizeof(typename
  // Graph::edge_data_type) * 2;
  Galois::preAlloc(numThreads +
                   approxNodeData / Galois::Runtime::pagePoolSize());
  Galois::reportPageAlloc("MeminfoPre");

  std::cout << "Running Asynch with CAS version\n";
  std::cout << "INFO: Using delta-step of " << (1 << stepShift) << "\n";
  std::cout << "WARNING: Performance varies considerably due to delta parameter.\n";
  std::cout << "WARNING: Do not expect the default to be good for your graph.\n";
  Galois::do_all_local(graph, 
                       [&graph] (GNode n) { graph.getData(n) = DIST_INFINITY; });
  graph.getData(source) = 0;
  Galois::StatTimer Tmain;
  Tmain.start();

  using namespace Galois::WorkList;
  typedef dChunkedFIFO<64> dChunk;
  typedef OrderedByIntegerMetric<UpdateRequestIndexer,dChunk> OBIM;
  Galois::for_each(UpdateRequest{source, 0}, SSSP{graph}, Galois::wl<OBIM>(), Galois::does_not_need_aborts<>());
  Tmain.stop();
  T.stop();

  Galois::reportPageAlloc("MeminfoPost");
  Galois::Runtime::reportNumaAlloc("NumaPost");
  
  std::cout << "Node " << reportNode << " has distance "
            << graph.getData(report) << "\n";

  if (!skipVerify) {
    if (verify<false>(graph, source)) {
      std::cout << "Verification successful.\n";
    } else {
      GALOIS_DIE("Verification failed");
    }
  }

  if (trackWork) {
    delete BadWork;
    delete WLEmptyWork;
  }

  return 0;
}
