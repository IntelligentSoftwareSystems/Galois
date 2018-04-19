/** Page rank application -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
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
 */

#include "Lonestar/BoilerPlate.h"
#include "PageRank-constants.h"
#include "galois/Galois.h"
#include "galois/LargeArray.h"
#include "galois/PerThreadContainer.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "galois/runtime/Profile.h"

namespace cll = llvm::cl;
const char* desc =
    "Computes page ranks a la Page and Brin. This is a pull-style algorithm.";

constexpr static const unsigned CHUNK_SIZE = 4096;

struct LNode {
  PRTy value;
  uint32_t nout;
};

typedef galois::graphs::LC_CSR_Graph<LNode, void>::with_no_lockable<
    true>::type ::with_numa_alloc<true>::type Graph;
typedef typename Graph::GraphNode GNode;

void initNodeData(Graph& g) {
  galois::do_all(galois::iterate(g),
                 [&](const GNode& n) {
                   auto& data = g.getData(n, galois::MethodFlag::UNPROTECTED);
                   data.value = INIT_RESIDUAL;
                   data.nout  = 0;
                 },
                 galois::no_stats(), galois::loopname("initNodeData"));
}

// Computing outdegrees in the tranpose graph is equivalent to computing the
// indegrees in the original graph
void computeOutDeg(Graph& graph) {
  galois::StatTimer outDegreeTimer("computeOutDeg");
  outDegreeTimer.start();

  galois::LargeArray<std::atomic<size_t>> vec;
  vec.allocateInterleaved(graph.size());

  galois::do_all(galois::iterate(graph),
                 [&](const GNode& src) { vec.constructAt(src, 0ul); },
                 galois::no_stats(), galois::loopname("InitDegVec"));

  galois::do_all(galois::iterate(graph),
                 [&](const GNode& src) {
                   for (auto nbr : graph.edges(src)) {
                     GNode dst = graph.getEdgeDst(nbr);
                     vec[dst].fetch_add(1ul);
                   }
                 },
                 galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
                 galois::no_stats(), galois::loopname("ComputeDeg"));

  galois::do_all(galois::iterate(graph),
                 [&](const GNode& src) {
                   auto& srcData =
                       graph.getData(src, galois::MethodFlag::UNPROTECTED);
                   srcData.nout = vec[src];
                 },
                 galois::no_stats(), galois::loopname("CopyDeg"));

  outDegreeTimer.stop();
}

// PageRank pull topological
void computePageRank(Graph& graph) {
  unsigned int iteration = 0;
  galois::GReduceMax<float> max_delta;

  while (true) {

    galois::do_all(galois::iterate(graph),
                   [&](const GNode& src) {
                     constexpr const galois::MethodFlag flag =
                         galois::MethodFlag::UNPROTECTED;

                     LNode& sdata = graph.getData(src, flag);
                     float sum    = 0.0;

                     for (auto jj = graph.edge_begin(src, flag),
                               ej = graph.edge_end(src, flag);
                          jj != ej; ++jj) {
                       GNode dst = graph.getEdgeDst(jj);

                       LNode& ddata = graph.getData(dst, flag);
                       sum += ddata.value / sdata.nout;
                     }

                     // New value of pagerank after computing contributions from
                     // incoming edges in the original graph
                     float value = sum * ALPHA + (1.0 - ALPHA);
                     // Find the delta in new and old pagerank values
                     float diff = std::fabs(value - sdata.value);

                     // Do not update pagerank before the diff is computed since
                     // there is a data dependence on the pagerank value
                     sdata.value = value;
                     max_delta.update(diff);
                   },
                   galois::no_stats(), galois::steal(),
                   galois::chunk_size<CHUNK_SIZE>(),
                   galois::loopname("PageRank"));

    float delta = max_delta.reduce();

#if DEBUG
    std::cout << "iteration: " << iteration << " max delta: " << delta << "\n";
#endif

    iteration += 1;
    if (delta <= tolerance || iteration >= maxIterations) {
      break;
    }
    max_delta.reset();

  } // end while(true)

  if (iteration >= maxIterations) {
    std::cerr << "ERROR: failed to converge in " << iteration << " iterations"
              << std::endl;
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer T("OverheadTime");
  T.start();

  Graph transposeGraph;

  std::cout << "Reading graph: " << filename << std::endl;
  std::cout << "WARNING: pull style algorithms work on the transpose of the "
               "actual graph "
            << std::endl;
  std::cout << "WARNING: this program assumes that " << filename
            << " contains transposed representation" << std::endl;

  galois::graphs::readGraph(transposeGraph, filename);
  std::cout << "Read " << transposeGraph.size() << " nodes, "
            << transposeGraph.sizeEdges() << " edges\n";

  galois::preAlloc(3 * numThreads + (3 * transposeGraph.size() *
                                     sizeof(typename Graph::node_data_type)) /
                                        galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  std::cout << "Running synchronous Pull version, tolerance:" << tolerance
            << ", maxIterations:" << maxIterations << "\n";

  initNodeData(transposeGraph);
  computeOutDeg(transposeGraph);

  galois::StatTimer Tmain("computePageRank");
  Tmain.start();
  computePageRank(transposeGraph);
  Tmain.stop();

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    printTop(transposeGraph);
  }

#if DEBUG
  printPageRank(transposeGraph);
#endif

  T.stop();

  return 0;
}
