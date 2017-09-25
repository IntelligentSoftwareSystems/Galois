/** Temporal path problems -*- C++ -*-
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
 * Earliest-arrival time on temporal graphs
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "galois/Galois.h"
#include "galois/Accumulator.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>

namespace cll = llvm::cl;

static const char* name = "Temporal Paths";
static const char* desc =
  "Computes the Earliest arrival time from a source node to all other nodes in a directed multi-graph";
static const char* url = "temporal_path";

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

typedef uint16_t timeTy;
struct time_dir {
  timeTy t;
  timeTy d;
};

typedef galois::graphs::LC_InlineEdge_Graph<std::atomic<timeTy>, time_dir> Graph;
typedef Graph::GraphNode GNode;

struct EarlyArivalTime {
  Graph& graph;
  EarlyArivalTime(Graph& g) : graph(g) {}
  void operator()(GNode src, galois::UserContext<GNode>& ctx) {
    const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
    auto& srcV = graph.getData(src, flag);
    
    for (auto ii : graph.edges(src, flag)) {
      GNode dst   = graph.getEdgeDst(ii);
      auto& dstV  = graph.getData(dst, flag);
      auto& edgeV = graph.getEdgeData(ii, flag);
      if (edgeV.t >= srcV) {
        auto newDist = edgeV.t + edgeV.d;
        auto oldDist = dstV.load(std::memory_order_relaxed);
        while (oldDist > newDist) {
          if (dstV.compare_exchange_weak(oldDist, newDist, std::memory_order_relaxed)) {
            ctx.push(dst);
          }
        }
      }
    }
  }
};

struct GNIndexer {
  Graph& g;
  GNIndexer(Graph& g) : g(g) {}
  unsigned long operator()(const GNode& val) const {
    return g.getData(val, galois::MethodFlag::UNPROTECTED).load(std::memory_order_relaxed) >> stepShift;
  }
};

int main(int argc, char** argv) {
  galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer T("OverheadTime");
  T.start();
  
  Graph graph;
  GNode source, report;

  galois::graphs::readGraph(graph, filename); 
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
  galois::preAlloc(numThreads +
                   approxNodeData / galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  uint16_t t = 0;
  for (auto n : graph) {
    for (auto e : graph.edges(n)) {
      graph.getEdgeData(e).t = ++t;
      graph.getEdgeData(e).d = 0; t % 3;
    }
  }

  std::cout << "Running Asynch with CAS version\n";
  std::cout << "INFO: Using delta-step of " << (1 << stepShift) << "\n";
  std::cout << "WARNING: Performance varies considerably due to delta parameter.\n";
  std::cout << "WARNING: Do not expect the default to be good for your graph.\n";
  galois::do_all_local(graph, 
                       [&graph] (GNode n) { graph.getData(n) = std::numeric_limits<timeTy>::max(); } );
  graph.getData(source) = 0;
  galois::StatTimer Tmain;
  Tmain.start();

  using namespace galois::worklists;
  typedef dChunkedFIFO<64> dChunk;
  typedef OrderedByIntegerMetric<GNIndexer,dChunk> OBIM;
  galois::for_each(source, EarlyArivalTime{graph}, galois::wl<OBIM>(GNIndexer{graph}), galois::no_conflicts());
  Tmain.stop();
  T.stop();

  galois::reportPageAlloc("MeminfoPost");
  galois::runtime::reportNumaAlloc("NumaPost");
  
  std::cout << "Node " << reportNode << " has earliest time "
            << graph.getData(report) << "\n";

  return 0;
}
