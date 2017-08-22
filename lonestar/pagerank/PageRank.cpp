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
 * @author Joyce Whang <joyce@cs.utexas.edu>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */


#include "Galois/Galois.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Graphs/TypeTraits.h"
#include "Lonestar/BoilerPlate.h"


#include <atomic>
#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>


namespace cll = llvm::cl;

static const char* name = "Page Rank";
static const char* desc = "Computes page ranks a la Page and Brin";
static const char* url = 0;

cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"), cll::init(0.01));

static const float alpha = 0.85;
typedef double PRTy;


struct LNode {
  PRTy value;
  std::atomic<PRTy> residual;
  // Initialize pagerank values
  void init() { value = 1.0 - alpha; residual = 0.0; }
  friend std::ostream& operator<<(std::ostream& os, const LNode& n) {
    os << "{PR " << n.value << ", residual " << n.residual << "}";
    return os;
  }
};


typedef Galois::Graph::LC_CSR_Graph<LNode,void>::with_numa_alloc<true>::type Graph;
typedef typename Graph::GraphNode GNode;

//! Make values unique
template<typename GNode>
struct TopPair {
  float value;
  GNode id;

  TopPair(float v, GNode i): value(v), id(i) { }

  bool operator<(const TopPair& b) const {
    if (value == b.value)
      return id > b.id;
    return value < b.value;
  }
};



template<typename Graph>
static void printTop(Graph& graph, int topn, const char *algo_name, int numThreads) {
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::node_data_reference node_data_reference;
  typedef TopPair<GNode> Pair;
  typedef std::map<Pair,GNode> Top;

  // normalize the PageRank value so that the sum is equal to one
  float sum=0;
  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    node_data_reference n = graph.getData(src);
    sum += n.value;
  }

  Top top;

  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src = *ii;
    node_data_reference n = graph.getData(src);
    float value = n.value / sum; // normalized PR (divide PR by sum)
    Pair key(value, src);

    if ((int) top.size() < topn) {
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
  for (typename Top::reverse_iterator ii = top.rbegin(), ei = top.rend(); ii != ei; ++ii, ++rank) {
    std::cout << rank << ": " << ii->first.value << " " << ii->first.id << "\n";
  }
}

PRTy atomicAdd(std::atomic<PRTy>& v, PRTy delta) {
  PRTy old;
  do {
    old = v;
  } while (!v.compare_exchange_strong(old, old + delta));
  return old;
}


struct PageRank {
  Graph& graph;
  PRTy tolerance;

  PageRank(Graph& g, PRTy t) : graph(g), tolerance(t) {}

  void operator()(const GNode& src, Galois::UserContext<GNode>& ctx) const {
    LNode& sdata = graph.getData(src);
    Galois::MethodFlag flag = Galois::MethodFlag::UNPROTECTED;

    if (std::fabs(sdata.residual) > tolerance) {
      PRTy oldResidual = sdata.residual.exchange(0.0);
      sdata.value += oldResidual;
      int src_nout = std::distance(graph.edge_begin(src, flag), graph.edge_end(src,flag));
      PRTy delta = oldResidual*alpha/src_nout;
      // for each out-going neighbors
      for (auto jj : graph.edges(src, flag)) {
        GNode dst = graph.getEdgeDst(jj);
        LNode& ddata = graph.getData(dst, flag);
        auto old = atomicAdd(ddata.residual, delta);
        if (std::fabs(old) <= tolerance && std::fabs(old + delta) >= tolerance)
          ctx.push(dst);
      }
    } else { // might need to reschedule self. But why?
      // ctx.push(src);
    }
  }
};


void initResidual(Graph& graph) {
  //use residual for the partial, scaled initial residual
  Galois::do_all_local(graph,
                       [&graph] (const typename Graph::GraphNode& src) {
                         //contribute residual
                         auto nout = std::distance(graph.edge_begin(src), graph.edge_end(src));
                         for (auto ii : graph.edges(src)) {
                           auto dst = graph.getEdgeDst(ii);
                           auto& ddata = graph.getData(dst);
                           atomicAdd(ddata.residual, 1.0/nout);
                         }
                       }, Galois::do_all_steal<true>());
  //scale residual
  Galois::do_all_local(graph,
                       [&graph] (const typename Graph::GraphNode& src) {
                         auto& data = graph.getData(src);
                         data.residual = data.residual * alpha * (1.0-alpha);
                       },
                       Galois::do_all_steal<true>());
}

int main(int argc, char **argv) {
  LonestarStart(argc, argv, name, desc, url);
  Galois::StatManager statManager;

  Galois::StatTimer T("OverheadTime");
  T.start();

  Graph graph;

  Galois::Graph::readGraph(graph, filename);

  std::cout << "Read " << std::distance(graph.begin(), graph.end()) << " Nodes\n";

  Galois::preAlloc(numThreads + (2*graph.size() * sizeof(typename Graph::node_data_type)) / Galois::Runtime::pagePoolSize());
  Galois::reportPageAlloc("MeminfoPre");

  std::cout << "Running Edge Async version\n";
  std::cout << "tolerance: " << tolerance << "\n";
  Galois::do_all_local(graph, [&graph] (typename Graph::GraphNode n) { graph.getData(n).init(); });
  initResidual(graph);
  typedef Galois::WorkList::dChunkedFIFO<256> WL;
  PageRank pr(graph, tolerance);

  Galois::StatTimer Tmain;
  Tmain.start();
  Galois::for_each_local(graph,
                         [&pr] (GNode src, auto& ctx) {
                           pr(src, ctx);
                         },
                         Galois::wl<WL>(),
                         Galois::loopname("Lonestar PageRank For Each"));
  Tmain.stop();

  Galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    printTop(graph, 10, "EdgeAsync", numThreads);
  }

  T.stop();

  return 0;
}
