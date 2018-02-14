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
 * Compute pageRank Pull version using residual on distributed Galois.
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

void initNodeData(Graph& g, galois::LargeArray<PRTy>& delta,
                  galois::LargeArray<std::atomic<PRTy>>& residual) {
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

void computeOutDeg(Graph& graph, galois::LargeArray<PRTy>& delta,
                   galois::LargeArray<std::atomic<PRTy>>& residual) {
  galois::StatTimer t("computeOutDeg");
  t.start();

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

  t.stop();
}

PRTy atomicAdd(std::atomic<PRTy>& v, PRTy delta) {
  PRTy old;
  do {
    old = v;
  } while (!v.compare_exchange_strong(old, old + delta));
  return old;
}

void computePageRankResidual(Graph& graph, galois::LargeArray<PRTy>& delta,
                             galois::LargeArray<std::atomic<PRTy>>& residual) {
  unsigned int iterations = 0;
  galois::GAccumulator<unsigned int> accum;

  while (true) {
    galois::do_all(galois::iterate(graph),
                   [&](const GNode& src) {
                     auto& sdata = graph.getData(src);
                     delta[src]  = 0;

                     if (std::fabs(residual[src]) > tolerance) {
                       PRTy oldResidual = residual[src].exchange(0.0);
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
                       atomicAdd(residual[src], sum);
                     }
                   },
                   galois::steal(), galois::no_stats(),
                   galois::loopname("PageRank"));

    iterations++;

    if (iterations >= maxIterations || !accum.reduce()) {
      break;
    }
    accum.reset();
  } // end while(true)

  if (iterations >= maxIterations) {
    std::cout << "Failed to converge\n";
  }
}

// Gets various values from the pageranks values/residuals of the graph
// struct PageRankSanity {
//   cll::opt<float>& local_tolerance;
//   Graph* graph;

//   galois::GAccumulator<float>& GAccumulator_sum;
//   galois::GAccumulator<float>& GAccumulator_sum_residual;
//   galois::GAccumulator<uint64_t>& GAccumulator_residual_over_tolerance;

//   galois::GReduceMax<float>& max_value;
//   galois::GReduceMin<float>& min_value;
//   galois::GReduceMax<float>& max_residual;
//   galois::GReduceMin<float>& min_residual;

//   PageRankSanity(
//       cll::opt<float>& _local_tolerance, Graph* _graph,
//       galois::GAccumulator<float>& _GAccumulator_sum,
//       galois::GAccumulator<float>& _GAccumulator_sum_residual,
//       galois::GAccumulator<uint64_t>& _GAccumulator_residual_over_tolerance,
//       galois::GReduceMax<float>& _max_value,
//       galois::GReduceMin<float>& _min_value,
//       galois::GReduceMax<float>& _max_residual,
//       galois::GReduceMin<float>& _min_residual)
//       : local_tolerance(_local_tolerance), graph(_graph),
//         GAccumulator_sum(_GAccumulator_sum),
//         GAccumulator_sum_residual(_GAccumulator_sum_residual),
//         GAccumulator_residual_over_tolerance(
//             _GAccumulator_residual_over_tolerance),
//         max_value(_max_value), min_value(_min_value),
//         max_residual(_max_residual), min_residual(_min_residual) {}

//   void static go(Graph& _graph, galois::GAccumulator<float>& DGA_sum,
//                  galois::GAccumulator<float>& DGA_sum_residual,
//                  galois::GAccumulator<uint64_t>& DGA_residual_over_tolerance,
//                  galois::GReduceMax<float>& max_value,
//                  galois::GReduceMin<float>& min_value,
//                  galois::GReduceMax<float>& max_residual,
//                  galois::GReduceMin<float>& min_residual) {
//     DGA_sum.reset();
//     DGA_sum_residual.reset();
//     max_value.reset();
//     max_residual.reset();
//     min_value.reset();
//     min_residual.reset();
//     DGA_residual_over_tolerance.reset();

//     {
//       galois::do_all(galois::iterate(_graph.masterNodesRange().begin(),
//                                      _graph.masterNodesRange().end()),
//                      PageRankSanity(tolerance, &_graph, DGA_sum,
//                                     DGA_sum_residual,
//                                     DGA_residual_over_tolerance, max_value,
//                                     min_value, max_residual, min_residual),
//                      galois::no_stats(), galois::loopname("PageRankSanity"));
//     }

//     float max_rank          = max_value.reduce();
//     float min_rank          = min_value.reduce();
//     float rank_sum          = DGA_sum.reduce();
//     float residual_sum      = DGA_sum_residual.reduce();
//     uint64_t over_tolerance = DGA_residual_over_tolerance.reduce();
//     float max_res           = max_residual.reduce();
//     float min_res           = min_residual.reduce();

//     galois::gPrint("Max rank is ", max_rank, "\n");
//     galois::gPrint("Min rank is ", min_rank, "\n");
//     galois::gPrint("Rank sum is ", rank_sum, "\n");
//     galois::gPrint("Residual sum is ", residual_sum, "\n");
//     galois::gPrint("# nodes with residual over ", tolerance, " (tolerance) is
//     ",
//                    over_tolerance, "\n");
//     galois::gPrint("Max residual is ", max_res, "\n");
//     galois::gPrint("Min residual is ", min_res, "\n");
//   }

//   /* Gets the max, min rank from all owned nodes and
//    * also the sum of ranks */
//   void operator()(GNode src) const {
//     NodeData& sdata = graph->getData(src);

//     max_value.update(sdata.value);
//     min_value.update(sdata.value);
//     max_residual.update(residual[src]);
//     min_residual.update(residual[src]);

//     GAccumulator_sum += sdata.value;
//     GAccumulator_sum_residual += residual[src];

//     if (residual[src] > local_tolerance) {
//       GAccumulator_residual_over_tolerance += 1;
//     }
//   }
// };

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

  galois::LargeArray<PRTy> delta;
  delta.allocateInterleaved(transposeGraph.size());
  galois::LargeArray<std::atomic<PRTy>> residual;
  residual.allocateInterleaved(transposeGraph.size());

  galois::preAlloc(numThreads + (3 * transposeGraph.size() *
                                 sizeof(typename Graph::node_data_type)) /
                                    galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  std::cout << "Running Pull residual version, tolerance:" << tolerance
            << ", maxIterations:" << maxIterations << "\n";

  initNodeData(transposeGraph, delta, residual);
  computeOutDeg(transposeGraph, delta, residual);

  galois::GAccumulator<unsigned int> PageRank_accum;
  galois::GAccumulator<float> DGA_sum;
  galois::GAccumulator<float> DGA_sum_residual;
  galois::GAccumulator<uint64_t> DGA_residual_over_tolerance;
  galois::GReduceMax<float> max_value;
  galois::GReduceMin<float> min_value;
  galois::GReduceMax<float> max_residual;
  galois::GReduceMin<float> min_residual;

  galois::StatTimer prTimer("PageRank");
  prTimer.start();
  computePageRankResidual(transposeGraph, delta, residual);
  prTimer.stop();

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    printTop(transposeGraph, 10);
  }

  // // sanity check
  // PageRankSanity::go(*hg, DGA_sum, DGA_sum_residual,
  //                    DGA_residual_over_tolerance, max_value, min_value,
  //                    max_residual, min_residual);

  // // Verify
  // if (verify) {
  //   for (auto ii = (*hg).masterNodesRange().begin();
  //        ii != (*hg).masterNodesRange().end(); ++ii) {
  //     galois::runtime::printOutput("% %\n", (*hg).getGID(*ii),
  //                                  (*hg).getData(*ii).value);
  //   }
  // }

  overheadTime.stop();
  return 0;
}
