/** Page rank application -*- C++ -*-
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
 */

#include "Lonestar/BoilerPlate.h"
#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"

#include <atomic>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>

namespace cll = llvm::cl;

const char* name = "Page Rank";
const char* desc =
    "Computes page ranks a la Page and Brin. This is a pull-style algorithm.";
const char* url = 0;

cll::opt<std::string> filename(cll::Positional,
                               cll::desc("<tranpose of input graph>"),
                               cll::Required);
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"),
                                 cll::init(0.000001));
cll::opt<unsigned int> maxIterations("maxIterations",
                                     cll::desc("Maximum iterations"),
                                     cll::init(10000000));

static const float alpha = (1 - 0.85);
typedef double PRTy;

struct LNode {
  PRTy oldValue;
  PRTy newValue;
  // Compute the outdegree in the original graph and save it to scale the
  // pagerank contribution
  std::atomic<uint32_t> nout;
};

typedef galois::graphs::LC_CSR_Graph<LNode, void>::with_numa_alloc<true>::type
    Graph;
typedef typename Graph::GraphNode GNode;

//! Make values unique
template <typename GNode>
struct TopPair {
  float value;
  GNode id;

  TopPair(float v, GNode i) : value(v), id(i) {}

  bool operator<(const TopPair& b) const {
    if (value == b.value)
      return id > b.id;
    return value < b.value;
  }
};

template <typename Graph>
static void printTop(Graph& graph, int topn, const char* algo_name,
                     int numThreads) {
  typedef typename Graph::node_data_reference node_data_reference;

  // normalize the PageRank value so that the sum is equal to one
  float sum = 0;
  std::ofstream out(std::string("out_" + std::to_string(numThreads)));
  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    node_data_reference n = graph.getData(*ii);
    out << *ii << " " << n.value << "\n";
#if 0
    GNode src             = *ii;
    node_data_reference n = graph.getData(src);
    sum += n.value;
#endif
  }
  out.close();

#if 0
  Top top;

  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {

    GNode src             = *ii;
    node_data_reference n = graph.getData(src);
    float value           = n.value / sum; // normalized PR (divide PR by sum)
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
#endif
}

uint32_t atomicAdd(std::atomic<uint32_t>& v, uint32_t delta) {
  uint32_t old;
  do {
    old = v;
  } while (!v.compare_exchange_strong(old, old + delta));
  return old;
}

/* Initialize all fields to 0 except for value which is (1 - alpha) */
struct InitGraph {
  void run(Graph& graph) {
    galois::do_all(galois::iterate(graph),
                   [&](const GNode src) {
                     auto& srcData    = graph.getData(src);
                     srcData.oldValue = alpha;
                     srcData.newValue = alpha;
                     srcData.nout     = 0;
                   },
                   galois::no_stats(), galois::loopname("InitGraph"));
  }
};

struct ComputeInDegrees {
  void run(Graph& graph) {
    galois::do_all(galois::iterate(graph),
                   [&](const GNode src) {
                     for (auto nbr : graph.edges(src)) {
                       GNode dst   = graph.getEdgeDst(nbr);
                       auto& ddata = graph.getData(dst);
                       atomicAdd(ddata.nout, (uint32_t)1);
                     }
                   },
                   galois::steal(), galois::no_stats(),
                   galois::loopname("ComputeInDegrees"));
  }
};

struct PageRank {

  void run(Graph& graph) {
    unsigned int iteration = 0;
    galois::GReduceMax<float> max_delta;

    while (true) {
      galois::do_all(galois::iterate(graph),
                     [&](const GNode& src) {
                       LNode& sdata = graph.getData(src);
                       constexpr const galois::MethodFlag flag =
                           galois::MethodFlag::UNPROTECTED;

                       PRTy sum = 0;
                       for (auto nbr : graph.edges(src)) {
                         GNode dst   = graph.getEdgeDst(nbr);
                         auto& ddata = graph.getData(dst);

                         if (ddata.nout > 0) {
                           PRTy contrib =
                               ddata.oldValue * (1 - alpha) / ddata.nout;
                           sum += contrib;
                         }
                       }

                       PRTy diff = sum - sdata.newValue;

                       if (std::fabs(diff) > tolerance) {
                         sdata.newValue += std::fabs(diff);
                         max_delta.update(std::fabs(diff));
                         galois::gPrint("diff : ", diff, "\n");
                       }
                     },
                     galois::loopname("PageRank"), galois::no_conflicts());

      galois::do_all(galois::iterate(graph),
                     [&](const GNode& src) {
                       LNode& sdata = graph.getData(src);
                       constexpr const galois::MethodFlag flag =
                           galois::MethodFlag::UNPROTECTED;
                       sdata.oldValue = sdata.newValue;
                     },
                     galois::loopname("Swap-pagerank"), galois::no_conflicts());
      iteration += 1;
#if 0
      if (iteration >= maxIterations) {
        galois::gPrint("Iter : ", iteration, "\n");
        break;
      }
#endif
      float delta = max_delta.reduce();
      if (delta < tolerance || iteration >= maxIterations) {
        galois::gPrint("Iter : ", iteration, "\n");
        break;
      }
      max_delta.reset();
    }
  }

  // void operator()(GNode src) const {
  //   auto& sdata = graph->getData(src);

  //   float sum = 0;
  //   for (auto nbr : graph->edges(src)) {
  //     GNode dst   = graph->getEdgeDst(nbr);
  //     auto& ddata = graph->getData(dst);

  //     if (ddata.nout > 0) {
  //       unsigned contrib = ddata.value*(1 - alpha)/ddata.nout;
  //       galois::add(sum, contrib);
  //     }
  //   }
  //   if (sum > tolerance) {
  //     sdata.value += sum;
  //   }
  // }
};

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer T("OverheadTime");
  T.start();

  Graph transposeGraph;
  galois::graphs::readGraph(transposeGraph, filename);

  std::cout << "Read "
            << std::distance(transposeGraph.begin(), transposeGraph.end())
            << " Nodes\n";

  galois::preAlloc(numThreads + (2 * transposeGraph.size() *
                                 sizeof(typename Graph::node_data_type)) /
                                    galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  std::cout << "Running Edge Async push version, tolerance: " << tolerance
            << "\n";

  galois::StatTimer outDegree;
  outDegree.start();
  InitGraph rg;
  rg.run(transposeGraph);
  ComputeInDegrees cdg;
  cdg.run(transposeGraph);
  outDegree.stop();

  galois::StatTimer Tmain;
  Tmain.start();
  PageRank p;
  p.run(transposeGraph);
  Tmain.stop();

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    printTop(transposeGraph, 10, "EdgeAsync", numThreads);
  }

  T.stop();

  return 0;
}
