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

cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"),
                               cll::Required);
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"),
                                 cll::init(0.000001));
cll::opt<unsigned int> maxIterations("maxIterations",
                                     cll::desc("Maximum iterations"),
                                     cll::init(10000000));

static const float alpha = 0.85;
typedef double PRTy;

struct LNode {
  PRTy value;
  PRTy residual;
  // Initialize pagerank values
  void init() {
    value    = 1.0 - alpha;
    residual = 0.0;
  }
  friend std::ostream& operator<<(std::ostream& os, const LNode& n) {
    os << "{PR " << n.value << ", residual " << n.residual << "}";
    return os;
  }
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
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::node_data_reference node_data_reference;
  typedef TopPair<GNode> Pair;
  typedef std::map<Pair, GNode> Top;

  // normalize the PageRank value so that the sum is equal to one
  float sum = 0;
  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src             = *ii;
    node_data_reference n = graph.getData(src);
    sum += n.value;
  }

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
}

void initResidual(Graph& graph) {
  // use residual for the partial, scaled initial residual
  galois::do_all(galois::iterate(graph),
                 [&graph](const typename Graph::GraphNode& src) {
                   auto& srcData = graph.getData(src);
                   constexpr const galois::MethodFlag flag =
                       galois::MethodFlag::UNPROTECTED;

                   // Compute residual from neighbors
                   for (auto ii : graph.edges(src, flag)) {
                     auto dst             = graph.getEdgeDst(ii);
                     auto dstNumNeighbors = std::distance(graph.edge_begin(dst),
                                                          graph.edge_end(dst));
                     srcData.residual += (1.0 / dstNumNeighbors);
                   }
                 },
                 galois::loopname("init-res-0"), galois::steal());
  // scale residual
  galois::do_all(galois::iterate(graph),
                 [&graph](const typename Graph::GraphNode& src) {
                   auto& data    = graph.getData(src);
                   data.residual = data.residual * alpha * (1.0 - alpha);
                 },
                 galois::loopname("init-res-1"), galois::steal());
}

struct PageRank {
  Graph& m_graph;
  PageRank(Graph& graph) : m_graph(graph) {}

  void operator()(Graph& graph) {
    unsigned int iteration = 0;
    galois::GReduceMax<float> max_delta;
    while (true) {
      galois::do_all(galois::iterate(graph),
                     [&](const GNode& src) {
                       LNode& sdata = graph.getData(src);
                       constexpr const galois::MethodFlag flag =
                           galois::MethodFlag::UNPROTECTED;

                       if (std::fabs(sdata.residual) > tolerance) {
                         sdata.value += sdata.residual;
                         // for each out-going neighbors
                         for (auto jj : graph.edges(src, flag)) {
                           GNode dst            = graph.getEdgeDst(jj);
                           LNode& ddata         = graph.getData(dst, flag);
                           auto dstNumNeighbors = std::distance(
                               graph.edge_begin(dst), graph.edge_end(dst));
                           PRTy delta =
                               ddata.residual * alpha / dstNumNeighbors;
                           sdata.residual += delta;
                           max_delta.update(delta);
                         }
                       }
                     },
                     galois::loopname("Main"), galois::no_conflicts());
      iteration += 1;
      float delta = max_delta.reduce();
      if (delta < tolerance || iteration >= maxIterations) {
        break;
      }
    }
  }
};

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer T("OverheadTime");
  T.start();

  Graph graph;
  galois::graphs::readGraph(graph, filename);

  std::cout << "Read " << std::distance(graph.begin(), graph.end())
            << " Nodes\n";

  galois::preAlloc(numThreads +
                   (2 * graph.size() * sizeof(typename Graph::node_data_type)) /
                       galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  std::cout << "Running Edge Async push version, tolerance: " << tolerance
            << "\n";

  galois::do_all(galois::iterate(graph), [&graph](typename Graph::GraphNode n) {
    graph.getData(n).init();
  });
  std::cout << "After init" << std::endl;
  initResidual(graph);
  std::cout << "Done till this point\n" << std::endl;
  galois::StatTimer Tmain;
  Tmain.start();
  PageRank p(graph);
  Tmain.stop();

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    printTop(graph, 10, "EdgeAsync", numThreads);
  }

  T.stop();

  return 0;
}
