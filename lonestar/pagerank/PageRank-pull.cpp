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

#define DEBUG 0

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
unsigned int iteration   = 0;

struct LNode {
  float value[2];
  // Compute the out degrees in the original graph
  std::atomic<uint32_t> nout;

  float getPageRank() { return value[1]; }
  float getPageRank(unsigned int it) { return value[it & 1]; }
  void setPageRank(unsigned it, float v) { value[(it + 1) & 1] = v; }
};

typedef galois::graphs::LC_CSR_Graph<LNode, void>::with_numa_alloc<true>::type
    Graph;
typedef typename Graph::GraphNode GNode;

uint32_t atomicAdd(std::atomic<uint32_t>& v, uint32_t delta) {
  uint32_t old;
  do {
    old = v;
  } while (!v.compare_exchange_strong(old, old + delta));
  return old;
}

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

struct Initialize {
  Graph& g;
  Initialize(Graph& g) : g(g) {}

  static void go(Graph& graph) {
    galois::do_all(galois::iterate(graph), Initialize(graph),
                   galois::no_stats(), galois::loopname("Initialize"));
  }

  void operator()(GNode n) const {
    LNode& data   = g.getData(n, galois::MethodFlag::UNPROTECTED);
    data.value[0] = 1.0;
    data.value[1] = 1.0;
    data.nout     = 0;
  }
};

struct Copy {
  Graph& g;

  Copy(Graph& g) : g(g) {}

  static void go(Graph& graph) {
    galois::do_all(galois::iterate(graph), Copy(graph), galois::no_stats(),
                   galois::loopname("Copy"));
  }

  void operator()(GNode n) const {
    LNode& data   = g.getData(n, galois::MethodFlag::UNPROTECTED);
    data.value[1] = data.value[0];
  }
};

struct PageRank {

  Graph& graph;
  galois::GReduceMax<double>& max_delta;

  PageRank(Graph& _graph, galois::GReduceMax<double>& _max_delta)
      : graph(_graph), max_delta(_max_delta) {}

  static void go(Graph& graph) {
    galois::GReduceMax<double> max_delta;

    while (true) {
      galois::do_all(galois::iterate(graph), PageRank(graph, max_delta),
                     galois::no_stats(), galois::loopname("PageRank"));

      float delta = max_delta.reduce();

#if DEBUG
      std::cout << "iteration: " << iteration << " max delta: " << delta
                << " small delta: " << sdelta << " ("
                << sdelta / (float)graph.size() << ")"
                << "\n";
#endif

      iteration += 1;
      if (delta <= tolerance || iteration >= maxIterations)
        break;
      max_delta.reset();
    }

    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }

    if (iteration & 1) {
      // Result already in right place
    } else {
      Copy::go(graph);
    }
  }

  void operator()(GNode src) const {
    LNode& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
    double sum   = 0;

    for (auto jj = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
              ej = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
         jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);

      LNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
      sum += ddata.getPageRank(iteration) / ddata.nout;
    }

    float value = sum * (1.0 - alpha) + alpha;
    float diff  = std::fabs(value - sdata.getPageRank(iteration));

    max_delta.update(diff);
    sdata.setPageRank(iteration, value);
  }
};

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
static void printTop(Graph& graph, int topn) {
  typedef typename Graph::node_data_reference node_data_reference;
  typedef TopPair<GNode> Pair;
  typedef std::map<Pair, GNode> Top;

  Top top;

  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src             = *ii;
    node_data_reference n = graph.getData(src);
    float value           = n.getPageRank();
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

  galois::StatTimer Tmain;
  Tmain.start();
  Initialize::go(transposeGraph);
  ComputeInDegrees cdg;
  cdg.run(transposeGraph);
  PageRank::go(transposeGraph);
  Tmain.stop();

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    printTop(transposeGraph, 10);
  }

  T.stop();

  return 0;
}
