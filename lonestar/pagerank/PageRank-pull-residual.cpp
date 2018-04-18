/** Residual based Page Rank -*- C++ -*-
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
 * Compute pageRank Pull version using residual.
 *
 */

#include "Lonestar/BoilerPlate.h"
#include "PageRank-constants.h"
#include "galois/Galois.h"
#include "galois/LargeArray.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "galois/gstl.h"

const char* desc =
    "Computes page ranks a la Page and Brin. This is a pull-style algorithm.";

constexpr static const unsigned CHUNK_SIZE = 32;

struct LNode {
  PRTy value;
  uint32_t nout;
};

typedef galois::graphs::LC_CSR_Graph<LNode, void>::with_no_lockable<
    true>::type ::with_numa_alloc<true>::type Graph;
typedef typename Graph::GraphNode GNode;

using DeltaArray    = galois::LargeArray<PRTy>;
using ResidualArray = galois::LargeArray<PRTy>;

void initNodeData(Graph& g, DeltaArray& delta, ResidualArray& residual) {
  galois::do_all(galois::iterate(g),
                 [&](const GNode& n) {
                   auto& sdata = g.getData(n, galois::MethodFlag::UNPROTECTED);
                   sdata.value = 0;
                   sdata.nout  = 0;
                   delta[n]    = 0;
                   residual[n] = INIT_RESIDUAL;
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
                   };
                 },
                 galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
                 galois::no_stats(), galois::loopname("ComputeDeg"));

  galois::do_all(galois::iterate(graph),
                 [&](const GNode& src) {
                   auto& srcData = graph.getData(src);
                   srcData.nout  = vec[src];
                 },
                 galois::no_stats(), galois::loopname("CopyDeg"));

  outDegreeTimer.stop();
}

void computePageRankResidual(Graph& graph, DeltaArray& delta,
                             ResidualArray& residual) {
  unsigned int iterations = 0;
  galois::GAccumulator<unsigned int> accum;

  while (true) {
    galois::do_all(galois::iterate(graph),
                   [&](const GNode& src) {
                     auto& sdata = graph.getData(src);
                     delta[src]  = 0;

                     if (residual[src] > tolerance) {
                       PRTy oldResidual = residual[src];
                       residual[src]    = 0.0;
                       sdata.value += oldResidual;
                       if (sdata.nout > 0) {
                         delta[src] = oldResidual * ALPHA / sdata.nout;
                         accum += 1;
                       }
                     }
                   },
                   galois::no_stats(), galois::loopname("PageRank_delta"));

    galois::do_all(galois::iterate(graph),
                   [&](const GNode& src) {
                     float sum = 0;
                     for (auto nbr : graph.edges(src)) {
                       GNode dst = graph.getEdgeDst(nbr);
                       if (delta[dst] > 0) {
                         sum += delta[dst];
                       }
                     }
                     if (sum > 0) {
                       residual[src] = sum;
                     }
                   },
                   galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
                   galois::no_stats(), galois::loopname("PageRank"));

#if DEBUG
    std::cout << "iteration: " << iterations << "\n";
#endif

    iterations++;
    if (iterations >= maxIterations || !accum.reduce()) {
      break;
    }
    accum.reset();
  } // end while(true)

  if (iterations >= maxIterations) {
    std::cerr << "ERROR: failed to converge in " << iterations << " iterations"
              << std::endl;
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer overheadTime("OverheadTime");
  overheadTime.start();

  Graph transposeGraph;
  galois::graphs::readGraph(transposeGraph, filename);
  std::cout << "Reading graph: " << filename << std::endl;
  std::cout << "Read " << transposeGraph.size() << " nodes, "
            << transposeGraph.sizeEdges() << " edges\n";

  DeltaArray delta;
  delta.allocateInterleaved(transposeGraph.size());
  ResidualArray residual;
  residual.allocateInterleaved(transposeGraph.size());

  galois::preAlloc(numThreads + (3 * transposeGraph.size() *
                                 sizeof(typename Graph::node_data_type)) /
                                    galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  std::cout << "Running Pull residual version, tolerance:" << tolerance
            << ", maxIterations:" << maxIterations << "\n";

  initNodeData(transposeGraph, delta, residual);
  computeOutDeg(transposeGraph);

  galois::StatTimer prTimer("PageRankResidual");
  prTimer.start();
  computePageRankResidual(transposeGraph, delta, residual);
  prTimer.stop();

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    printTop(transposeGraph);
  }

#if DEBUG
  printPageRank(transposeGraph);
#endif

  overheadTime.stop();
  return 0;
}
