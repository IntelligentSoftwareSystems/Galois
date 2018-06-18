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

#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/graphs/LC_Dist_Graph.h"
#include "galois/Graph/FileGraph.h"
#include "galois/graphs/LC_Dist_InOut_Graph_withEdgeData.h"
#include "galois/Bag.h"
#include "galois/runtime/Context.h"

#include "Lonestar/BoilerPlate.h"

#include "galois/graphs/Graph3.h"

#include <mpi.h>
#include <iostream>
#include <typeinfo>
#include <algorithm>

static const char* const name = "Page Rank - Distributed";
static const char* const desc = "Computes PageRank on Distributed Galois";
static const char* const url  = 0;

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> inputFile_tr(cll::Positional,
                                          cll::desc("<transpose input file>"),
                                          cll::Required);

static cll::opt<unsigned int> maxIterations("maxIterations",
                                            cll::desc("Maximum iterations"),
                                            cll::init(2));

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

typedef galois::graphs::LC_Dist_InOut<LNode> Graph;
typedef typename Graph::GraphNode GNode;

typedef galois::graphs::ThirdGraph<LNode, void,
                                   galois::graphs::EdgeDirection::Out>
    Graph_3;

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

  void static go(Graph::pointer g) {
    galois::for_each(g, InitializeGraph{g}, galois::loopname("init"));
  }

  void static remoteUpdate(Graph::pointer pr, GNode src, float delta) {
    auto& lnode = pr->at(src, galois::MethodFlag::SRC_ONLY);
    atomicAdd(lnode.residual, delta);
  }

  void operator()(GNode src, galois::UserContext<GNode>& cnx) const {
    LNode& sdata = g->at(src);
    sdata.value  = 1.0 - alpha;
    sdata.nout = std::distance(g->edge_begin(src, galois::MethodFlag::SRC_ONLY),
                               g->edge_end(src, galois::MethodFlag::SRC_ONLY));
    sdata.residual = 0;

    const galois::MethodFlag lockflag = galois::MethodFlag::SRC_ONLY;
    auto& net = galois::runtime::getSystemNetworkInterface();
    if (sdata.nout > 0) {
      float delta = sdata.value * alpha / sdata.nout;
      // for each out-going neighbors
      for (auto jj = g->edge_begin(src, lockflag),
                ej = g->edge_end(src, lockflag);
           jj != ej; ++jj) {
        GNode dst = g->dst(jj);
        if (g->isLocal(dst)) {
          LNode& ddata = g->at(dst, lockflag);
          atomicAdd(ddata.residual, delta);
        } else {
          net.sendAlt(g->getHost(dst), remoteUpdate, g, dst, delta);
        }
      }
    }
  }

  typedef int tt_is_copyable;
};

struct checkGraph {
  Graph::pointer g;

  void static go(Graph::pointer g) {
    galois::for_each(g, checkGraph{g}, galois::loopname("checkGraph"));
  }

  void operator()(GNode n, galois::UserContext<GNode>& cnx) const {
    LNode& data = g->at(n);
    std::cout << data.value << "\n";
  }

  typedef int tt_is_copyable;
};

struct PageRank {
  Graph::pointer g;
  void static go(Graph::pointer g) {
    galois::Timer round_time;
    for (int iterations = 0; iterations < maxIterations; ++iterations) {
      std::cout << "Iteration : " << iterations << "  start : "
                << "\n";
      round_time.start();
      galois::for_each(g, PageRank{g}, galois::loopname("Page Rank"));
      round_time.stop();
      std::cout << "Iteration : " << iterations
                << "  Time : " << round_time.get() << " ms\n";
    }
  }

  void operator()(GNode src, galois::UserContext<GNode>& cnx) const {
    double sum                  = 0;
    galois::MethodFlag flag_src = galois::MethodFlag::SRC_ONLY;

    LNode& sdata = g->at(src, galois::MethodFlag::SRC_ONLY);
    for (auto jj = g->in_edge_begin(src, flag_src),
              ej = g->in_edge_end(src, flag_src);
         jj != ej; ++jj) {
      GNode dst    = g->dst(jj, flag_src);
      LNode& ddata = g->at(dst, flag_src);
      // std::cout << ddata.value << ", " << ddata.nout <<"\n";
      if (ddata.nout != 0) {
        sum += ddata.value / ddata.nout;
      }
    }
    float value = (1.0 - alpha) * sum + alpha;
    float diff  = std::fabs(value - sdata.value);
    if (diff > TOLERANCE) {
      sdata.value = value;
    }
  }

  typedef int tt_is_copyable;
};

struct PageRankMsg {
  Graph::pointer g;
  static int count;
  void static go(Graph::pointer g) {
    galois::Timer round_time;
    for (int iterations = 0; iterations < maxIterations; ++iterations) {
      std::cout << "Iteration : " << iterations << "  start : "
                << "\n";
      round_time.start();
      galois::for_each(g, PageRankMsg{g}, galois::loopname("Page Rank"));
      round_time.stop();
      std::cout << "Iteration : " << iterations
                << "  Time : " << round_time.get() << " ms\n";
      galois::runtime::getSystemNetworkInterface().dumpStats();
    }
  }

  void static remoteUpdate(Graph::pointer pr, GNode src, float delta) {
    auto& lnode = pr->at(src, galois::MethodFlag::SRC_ONLY);
    atomicAdd(lnode.residual, delta);
  }

  void operator()(GNode src, galois::UserContext<GNode>& cnx) const {
    LNode& sdata                = g->at(src);
    galois::MethodFlag lockflag = galois::MethodFlag::SRC_ONLY;
    auto& net                   = galois::runtime::getSystemNetworkInterface();
    /*
     * DEBUG
     */
    /* if(net.ID == 0)
       {
         count++;
         std::cout << "count : " << count << "\n";
       }
       */
    float oldResidual = sdata.residual.exchange(0.0);
    sdata.value       = sdata.value + oldResidual;
    if (sdata.nout > 0) {
      float delta = oldResidual * alpha / sdata.nout;
      // for each out-going neighbors
      for (auto jj = g->edge_begin(src, lockflag),
                ej = g->edge_end(src, lockflag);
           jj != ej; ++jj) {
        GNode dst = g->dst(jj);
        if (g->isLocal(dst)) {
          LNode& ddata = g->at(dst, lockflag);
          atomicAdd(ddata.residual, delta);
        } else {
          net.sendAlt(g->getHost(dst), remoteUpdate, g, dst, delta);
        }
      }
    }
  }
  typedef int tt_is_copyable;
};

int PageRankMsg::count = 0;

/*
 * collect page rank of all the nodes
 * */

float compute_total_rank(Graph::pointer g) {
  float total_rank = 0;

  for (auto ii = g->begin(), ei = g->end(); ii != ei; ++ii) {
    LNode& node = g->at(*ii);
    total_rank += node.value;
  }

  return total_rank;
}

#if 0
/* Compute Graph Distribution */
using std::cout;
using std::vector;
using namespace galois::runtime;

void compute_graph_distribution(Graph::pointer g) {
  int n = galois::runtime::getSystemNetworkInterface().Num;
  vector<vector<int>> dist_vec(n);
  vector<int> local_count(n,0);

  for(auto ii = g->begin(), ei = g->end(); ii != ei; ++ii) {
    //iterate over all the edges of a node
    fatPointer fptr = static_cast<fatPointer>(*ii);
    auto hostID_src = fptr.getHost();
    //cout << "src :" << hostID_src <<"\n";
    for(auto jj = g->in_edge_begin(*ii, galois::MethodFlag::SRC_ONLY), ej = g->in_edge_end(*ii, galois::MethodFlag::SRC_ONLY); jj != ej; ++jj) {

      GNode dst = g->dst(jj, galois::MethodFlag::SRC_ONLY);
      fatPointer fptr_dst = static_cast<fatPointer>(dst);
      auto hostID_dst = fptr_dst.getHost();
      if(hostID_dst == hostID_src) {
        ++local_count[hostID_src];
      } else {
        //cout << "\thostID_dst : " << hostID_dst << "\n";
        dist_vec[hostID_src].push_back(hostID_dst);
      }

    }
  }

  int total_edges = 0;
  /* print out stats */
  //local count
  cout << "local Count\n";
  int h = 0;
  for (auto v : local_count) {
    total_edges += v;
    cout << "Host : " << h << "\n";
    cout << "\t|E| "<< v <<"\n";
  }

  //remote
  cout << "remote count\n";
  for (int i = 0; i < n; ++i) {
    total_edges += dist_vec[i].size(); 
    cout <<"For : " << i <<" : " << dist_vec[i].size() << "\n";
  }


  cout << "Remote edge counts\n\n";
  using std::count;
  for (int i = 0; i < n; ++i) {
    for(int j = 0; j < n; ++j) {
      cout << "from : " << i << " to : " << j << " => " << count(dist_vec[i].begin(), dist_vec[i].end(), j) << "\n";
    }

    cout << "\n";
  }

  /* Percetage of local edges */
  cout << "Local Edges %\n\n";
  h = 0;
  for (auto v : local_count) {
    cout << "Host : " << h << "\n";
    cout << "\t|E| : "<< v << " % : " << (double)v*((double)total_edges/100.0)  <<"\n";
  }

  /* Percetage of remote edges */
  cout << "Remote Edges %\n\n";
  for (int i = 0; i < n; ++i) {
    cout << "Host : " << i << "\n";
    cout <<"\t|E| : " << dist_vec[i].size() << " % : "<< (double)dist_vec[i].size()*((double)total_edges/100.0) <<"\n";
  }


  cout << "TOTOAL EDGES in Graph : " << total_edges << "\n";

}
#endif

int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);
  galois::StatManager statManager;

  galois::Timer timerLoad;
  timerLoad.start();

  // allocate local computation graph and Reading from the inputFile using
  // FileGraph NOTE: We are computing in edges on the fly and then using then in
  // Graph construction. std::vector<unsigned> counts; std::vector<unsigned>
  // In_counts;
  Graph::pointer g = Graph::allocate(inputFile, inputFile_tr);

  /*

            galois::graphs::FileGraph fg;
              fg.fromFile(inputFile);

              std::cout << " filegraph : " << fg.size() << "\n";
              for(auto& N : fg)
              {
                counts.push_back(std::distance(fg.edge_begin(N),
     fg.edge_end(N))); for(auto ii = fg.edge_begin(N), ei = fg.edge_end(N); ii
     != ei; ++ii)
                {
                  unsigned dst = fg.getEdgeDst(ii);
                  if(dst >= In_counts.size()) {
                    In_counts.resize(dst+1);
                  }
                  In_counts[dst]+=1;
                }
              }
              if(counts.size() >  In_counts.size())
                In_counts.resize(counts.size());
  */

  // Graph Construction ENDS here.
  timerLoad.stop();
  std::cout << "Graph Loading: " << timerLoad.get() << " ms\n";
  galois::runtime::getSystemNetworkInterface().dumpStats();

  /*
   * Check Graph Loading
   *
   */

  // std::cout <<"CHECKING GRAPH : " << check_graph(g, counts, In_counts)
  // <<"\n";

  // Graph Initialization begins.
  galois::Timer timerInit;
  timerInit.start();

  InitializeGraph::go(g);

  timerInit.stop();
  std::cout << "Graph Initialization: " << timerInit.get() << " ms\n";
  galois::runtime::getSystemNetworkInterface().dumpStats();

  // g->prefetch_all();

  // std::cout << " All prefetched\n";

  galois::Timer timerPR;
  timerPR.start();

  // PageRank::go(g);
  PageRankMsg::go(g);

  timerPR.stop();

  std::cout << "Page Rank: " << timerPR.get() << " ms\n";
  galois::runtime::getSystemNetworkInterface().dumpStats();

  // HACK: prefetch all the nodes. For Blocking serial code.
  int nodes_check = 0;
  for (auto N = g->begin(); N != g->end(); ++N) {
    ++nodes_check;
    // galois::runtime::prefetch(*N);
  }
  std::cout << "Nodes_check = " << nodes_check << "\n";

  galois::runtime::getSystemNetworkInterface().dumpStats();

  std::cout << "Total Page Rank: " << compute_total_rank(g) << "\n";
  galois::runtime::getSystemNetworkInterface().dumpStats();

  // std::cout << "Computing graph Distribution\n";
  // compute_graph_distribution(g);

  galois::runtime::getSystemNetworkInterface().terminate();
  return 0;
}
