/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#include "Lonestar/BoilerPlate.h"
#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"

static const char* name = "Count levels";
static const char* desc = "Computes the number of degree levels";
static const char* url  = 0;

#define DEBUG false

namespace cll = llvm::cl;

static cll::opt<std::string>
    filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<unsigned int> startNode("startNode",
                                        cll::desc("Node to start search from"),
                                        cll::init(0));

enum COLOR { WHITE, GRAY, BLACK };

struct LNode {
  uint32_t dist;
  COLOR color;
};

using Graph = galois::graphs::LC_CSR_Graph<LNode, void>::with_numa_alloc<
    true>::type ::with_no_lockable<true>::type;
using GNode = Graph::GraphNode;

static const unsigned int DIST_INFINITY =
    std::numeric_limits<unsigned int>::max();

const galois::gstl::Vector<size_t>& countLevels(Graph& graph) {

  //! [Define GReducible]
  galois::GVectorPerItemReduce<size_t, std::plus<size_t>> reducer;

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    LNode srcdata = graph.getData(n);
    if (srcdata.dist == DIST_INFINITY) {
      return;
    }
    reducer.update(srcdata.dist, 1);
  });

  return reducer.reduce();
  //! [Define GReducible]
}

// constexpr static const unsigned CHUNK_SIZE = 16;

void bfsSerial(Graph& graph, GNode source) {
  constexpr galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;

  LNode& sdata = graph.getData(source, flag);
  sdata.dist   = 0u;
  sdata.color  = GRAY;

  std::queue<GNode> queue;
  queue.push(source);

  while (!queue.empty()) {
    GNode curr = queue.front();
    sdata      = graph.getData(curr, flag);
    queue.pop();

    // iterate over edges from node n
    for (auto e : graph.edges(curr)) {
      GNode dst    = graph.getEdgeDst(e);
      LNode& ddata = graph.getData(dst);

      if (ddata.color == WHITE) {
        ddata.color = GRAY;
        ddata.dist  = sdata.dist + 1;
        queue.push(dst);
      }
    }
    sdata.color = BLACK;
  } // end while
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer OT("OverheadTime");
  OT.start();

  Graph graph;
  galois::graphs::readGraph(graph, filename);
  std::cout << "Read " << graph.size() << " nodes, " << graph.sizeEdges()
            << " edges\n";

  galois::preAlloc(5 * numThreads +
                   (2 * graph.size() * sizeof(typename Graph::node_data_type)) /
                       galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  galois::do_all(galois::iterate(graph),
                 [&](const GNode& src) {
                   LNode& sdata = graph.getData(src);
                   sdata.color  = WHITE;
                   sdata.dist   = DIST_INFINITY;
                 },
                 galois::no_stats());

  if (startNode >= graph.size()) {
    std::cerr << "Source node index " << startNode
              << " is greater than the graph size" << graph.size()
              << ", failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }
  GNode source;
  auto it = graph.begin();
  std::advance(it, startNode);
  source = *it;

  galois::StatTimer T;
  T.start();
  bfsSerial(graph, source);
  const auto& counts = countLevels(graph);
  T.stop();

  galois::reportPageAlloc("MeminfoPost");

#if DEBUG
  for (auto n : graph) {
    LNode& data = graph.getData(n);
    std::cout << "Node: " << n << " BFS dist:" << data.dist << std::endl;
  }
#endif

  std::cout << "Number of BFS levels: " << counts.size() << "\n";

  OT.stop();

  return EXIT_SUCCESS;
}
