/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/graphs/LC_Dist_Graph.h"
#include "galois/Graph/FileGraph.h"
#include "galois/graphs/LC_Dist_InOut_Graph.h"
#include "galois/Bag.h"

#include "Lonestar/BoilerPlate.h"

#include <iostream>
#include <typeinfo>

static const char* const name = "Page Rank - Distributed";
static const char* const desc = "Computes PageRank on Distributed Galois";
static const char* const url  = 0;

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations",
                                            cll::desc("Maximum iterations"),
                                            cll::init(1));

static int TOTAL_NODES;

struct LNode {
  float value;
  std::atomic<float> residual;
  unsigned int nout;
  LNode() : value(1.0), nout(0) {}
  LNode(const LNode& rhs)
      : value(rhs.value), residual(rhs.residual.load()), nout(rhs.nout) {}
  float getPageRank() { return value; }

  typedef int tt_is_copyable;
};

typedef galois::graphs::LC_Dist_InOut<LNode, int> Graph;
typedef typename Graph::GraphNode GNode;

// Constants for page Rank Algo.
//! d is the damping factor. Alpha is the prob that user will do a random jump,
//! i.e., 1 - d
static const double alpha = (1.0 - 0.85);

//! maximum relative change until we deem convergence
static const double TOLERANCE = 0.1;

template <typename PRTy>
PRTy atomicAdd(std::atomic<PRTy>& v, PRTy delta) {
  PRTy old;
  do {
    old = v;
  } while (!v.compare_exchange_strong(old, old + delta));
  return old;
}

struct InitializeGraph {
  Graph::pointer g;
  //    InitializeGraph(Graph::pointer _g) : g(_g) { }

  void static go(Graph::pointer g) {
    galois::for_each(g, InitializeGraph{g}, galois::loopname("init"));
  }

  void operator()(GNode n, galois::UserContext<GNode>& cnx) const {
    LNode& data   = g->at(n);
    data.value    = 1.0 - alpha;
    data.residual = 0;
    // Adding galois::NONE is imp. here since we don't need any blocking locks.
    data.nout = std::distance(g->in_edge_begin(n, galois::MethodFlag::SRC_ONLY),
                              g->in_edge_end(n, galois::MethodFlag::SRC_ONLY));
  }

  typedef int tt_is_copyable;
};

struct PageRank {
  Graph::pointer g;
  void static go(Graph::pointer g) {
    galois::Timer round_time;
    for (int iterations = 0; iterations < maxIterations; ++iterations) {
      round_time.start();
      galois::for_each(g, PageRank{g}, galois::loopname("Page Rank"));
      round_time.stop();
      std::cout << "Iteration : " << iterations
                << "  Time : " << round_time.get() << "ms\n";
    }
  }

  void operator()(GNode src, galois::UserContext<GNode>& cnx) const {
    double sum   = 0;
    LNode& sdata = g->at(src);
    // std::cout << "n :" << n.nout <<"\n";
    for (auto jj = g->in_edge_begin(src, galois::MethodFlag::ALL),
              ej = g->in_edge_end(src, galois::MethodFlag::SRC_ONLY);
         jj != ej; ++jj) {
      GNode dst    = g->dst(jj, galois::MethodFlag::SRC_ONLY);
      LNode& ddata = g->at(dst, galois::MethodFlag::SRC_ONLY);
      sum += ddata.value / ddata.nout;
    }
    float value = (1.0 - alpha) * sum + alpha;
    float diff  = std::fabs(value - sdata.value);
    if (diff > TOLERANCE) {
      sdata.value = value;
      /*   for (auto jj = g->edge_begin(src, galois::MethodFlag::SRC_ONLY), ej =
         g->edge_end(src, galois::MethodFlag::SRC_ONLY); jj != ej; ++jj) { GNode
         dst = g->dst(jj, galois::MethodFlag::SRC_ONLY); cnx.push(dst);
           }
           */
    }
  }

  typedef int tt_is_copyable;
};

struct PageRankMsg {
  Graph::pointer g;
  void static go(Graph::pointer g) {
    galois::Timer round_time;
    for (int iterations = 0; iterations < maxIterations; ++iterations) {
      round_time.start();
      galois::for_each(g, PageRank{g}, galois::loopname("Page Rank"));
      round_time.stop();
      std::cout << "Iteration : " << iterations
                << "  Time : " << round_time.get() << "ms\n";
    }
  }

  void static remoteUpdate(Graph::pointer pr, GNode src, float delta) {
    auto& lnode = pr->at(src, galois::MethodFlag::NONE);
    atomicAdd(lnode.residual, delta);
  }

  void operator()(GNode src, galois::UserContext<GNode>& cnx) const {
    LNode& sdata                = g->at(src);
    galois::MethodFlag lockflag = galois::MethodFlag::NONE;

    float oldResidual = sdata.residual.exchange(0.0);
    sdata.value       = sdata.value + oldResidual;
    float delta       = oldResidual * alpha / sdata.nout;
    // for each out-going neighbors
    auto& net = galois::runtime::getSystemNetworkInterface();
    for (auto jj = g->edge_begin(src, lockflag),
              ej = g->edge_end(src, lockflag);
         jj != ej; ++jj) {
      GNode dst = g->dst(jj);
      if (dst.isLocal()) {
        LNode& ddata = g->at(dst, lockflag);
        atomicAdd(ddata.residual, delta);
      } else {
        net.sendAlt(((galois::runtime::fatPointer)dst).getHost(), remoteUpdate,
                    g, dst, delta);
      }
    }
  }

  typedef int tt_is_copyable;
};

/*
 * collect page rank of all the nodes
 *
 *
 */
int compute_total_rank(Graph::pointer g) {
  int total_rank = 0;

  for (auto ii = g->begin(), ei = g->end(); ii != ei; ++ii) {
    LNode& node = g->at(*ii);
    total_rank += node.value;
  }

  return total_rank;
}

int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);
  galois::StatManager statManager;

  galois::Timer timerLoad;
  timerLoad.start();

  /* Allocate local computation graph and Reading from the inputFile using
   *FileGraph NOTE: We are computing in edges on the fly and then using then in
   *Graph construction.
   *
   */
  Graph::pointer g;
  {
    galois::graphs::FileGraph fg;
    fg.fromFile(inputFile);
    std::vector<unsigned> counts;
    std::vector<unsigned> In_counts;
    for (auto& N : fg) {
      counts.push_back(std::distance(fg.edge_begin(N), fg.edge_end(N)));
      for (auto ii = fg.edge_begin(N), ei = fg.edge_end(N); ii != ei; ++ii) {
        unsigned dst = fg.getEdgeDst(ii);
        if (dst >= In_counts.size()) {
          /* NOTE:+1 is imp because vec.resize makes sure new vec can hold dst
           * entries so it will not have vec[dst] which is (dst+1)th entry!!
           *
           */
          In_counts.resize(dst + 1);
        }
        In_counts[dst] += 1;
      }
    }
    if (counts.size() > In_counts.size())
      In_counts.resize(counts.size());

    TOTAL_NODES = counts.size();

    std::cout << "size of transpose : " << In_counts.size()
              << " : : " << In_counts[0] << "\n";
    std::cout << "size of counts : " << counts.size() << "\n";
    g = Graph::allocate(counts, In_counts);

    // HACK: Prefetch all the nodes. For Blocking serial code.
    int nodes_check = 0;
    for (auto N = g->begin(); N != g->end(); ++N) {
      ++nodes_check;
      galois::runtime::prefetch(*N);
    }
    std::cout << "Nodes_check = " << nodes_check << "\n";

    for (unsigned x = 0; x < counts.size(); ++x) {
      auto fgn = *(fg.begin() + x);
      auto gn  = *(g->begin() + x);
      for (auto ii = fg.edge_begin(fgn), ee = fg.edge_end(fgn); ii != ee;
           ++ii) {
        unsigned dst = fg.getEdgeDst(ii);
        int val      = fg.getEdgeData<int>(ii);
        g->addEdge(gn, *(g->begin() + dst), val, galois::MethodFlag::SRC_ONLY);
        g->addInEdge(*(g->begin() + dst), gn, val,
                     galois::MethodFlag::SRC_ONLY);
      }
    }
  }
  // Graph Construction ENDS here.
  timerLoad.stop();
  std::cout << "Graph Loading: " << timerLoad.get() << " ms\n";

  // Graph Initialization begins.
  galois::Timer timerInit;
  timerInit.start();
  InitializeGraph::go(g);
  timerInit.stop();
  std::cout << "Graph Initialization: " << timerInit.get() << " ms\n";

  // PageRank begins.
  galois::Timer timerPR;
  timerPR.start();
  PageRankMsg::go(g);
  timerPR.stop();
  std::cout << "Page Rank: " << timerPR.get() << " ms\n";

  // HACK: prefetch all the nodes. For Blocking serial code.
  int nodes_check = 0;
  for (auto N = g->begin(); N != g->end(); ++N) {
    ++nodes_check;
    galois::runtime::prefetch(*N);
  }
  std::cout << "Nodes_check = " << nodes_check << "\n";
  std::cout << "Total Page Rank: " << compute_total_rank(g) << "\n";

  galois::runtime::getSystemNetworkInterface().terminate();
  return 0;
}
