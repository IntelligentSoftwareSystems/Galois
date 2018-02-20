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
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 * @author Roshan Dathathri <roshan@cs.utexas.edu>
 * @author Loc Hoang <l_hoang@utexas.edu> (sanity check operators)
 */

#include "Lonestar/BoilerPlate.h"
#include "PageRank-constants.h"
#include "galois/Galois.h"
#include "galois/LargeArray.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "galois/gstl.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

namespace cll = llvm::cl;
const char* desc =
    "Computes page ranks a la Page and Brin. This is a push-style algorithm.";

constexpr static const unsigned CHUNK_SIZE = 32;

// We require a transpose graph since this is a pull-style algorithm
static cll::opt<std::string> filename(cll::Positional,
                                      cll::desc("<tranpose of input graph>"),
                                      cll::Required);
static cll::opt<float> tolerance("tolerance",
                                 cll::desc("tolerance for residual"),
                                 cll::init(TOLERANCE));
static cll::opt<unsigned int>
    maxIterations("maxIterations",
                  cll::desc("Maximum iterations: Default 1000"),
                  cll::init(MAX_ITER));

struct LNode {
  PRTy value;
  std::atomic<uint32_t> nout;
};

typedef galois::graphs::LC_CSR_Graph<LNode, void>::with_no_lockable<
    true>::type ::with_numa_alloc<true>::type Graph;
typedef typename Graph::GraphNode GNode;

template <typename Graph>
static void printTop(Graph& graph, int topn) {
  typedef typename Graph::node_data_reference node_data_reference;
  typedef TopPair<GNode> Pair;
  typedef std::map<Pair, GNode> Top;

  Top top;

  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src             = *ii;
    node_data_reference n = graph.getData(src);
    PRTy value            = n.value;
    Pair key(value, src);

    if ((int)top.size() < topn) {
      top.insert(std::make_pair(key, src));
      continue;
    }

    if (top.begin()->first < key) {
      top.erase(top.begin());
      top.insert(std::make_pair(key, src));
    }
  }

  int rank = 1;
  std::cout << "Rank PageRank Id\n";
  for (typename Top::reverse_iterator ii = top.rbegin(), ei = top.rend();
       ii != ei; ++ii, ++rank) {
    std::cout << rank << ": " << ii->first.value << " " << ii->first.id << "\n";
  }
}

using DeltaArray    = galois::LargeArray<PRTy>;
using ResidualArray = galois::LargeArray<PRTy>;

void initNodeData(Graph& g, DeltaArray& delta, ResidualArray& residual) {
  galois::do_all(galois::iterate(g),
                 [&](const GNode& n) {
                   auto& sdata = g.getData(n, galois::MethodFlag::UNPROTECTED);
                   sdata.value = PR_INIT_VAL;
                   sdata.nout  = 0;
                   delta[n]    = 0;
                   residual[n] = ALPHA;
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
                     // This is equivalent to computing the outdegree in the
                     // original (not transpose) graph
                     vec[dst].fetch_add(1ul);
                   }
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
                   galois::steal(), galois::no_stats(),
                   galois::loopname("PageRank"));

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
    printTop(transposeGraph, PRINT_TOP);
  }

  overheadTime.stop();
  return 0;
}
