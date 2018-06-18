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
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "Lonestar/BoilerPlate.h"

#include <atomic>
#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

#include "PageRank.h"
#include "galois/worklists/WorkSet.h"
#include "galois/worklists/MarkingSet.h"

namespace cll = llvm::cl;

static const char* name = "Page Rank";
static const char* desc = "Computes page ranks a la Page and Brin";
static const char* url  = 0;

enum Algo {
  asyncB,
  asyncB_hset,
  asyncB_mset,
  asyncB_oset,
  asyncB_prt,
  asyncB_prt_hset,
  asyncB_prt_mset,
  asyncB_prt_oset
};

cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"),
                               cll::Required);
static cll::opt<std::string>
    transposeGraphName("graphTranspose", cll::desc("Transpose of input graph"));
cll::opt<unsigned int> maxIterations("maxIterations",
                                     cll::desc("Maximum iterations"),
                                     cll::init(10000000));
cll::opt<unsigned int>
    memoryLimit("memoryLimit",
                cll::desc("Memory limit for out-of-core algorithms (in MB)"),
                cll::init(~0U));
static cll::opt<float> amp("amp", cll::desc("amp for priority"),
                           cll::init(100));
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"),
                                 cll::init(0.01));
static cll::opt<bool> dbg("dbg", cll::desc("dbg"), cll::init(false));
static cll::opt<std::string> algo_str("algo_str", cll::desc("algo_str"),
                                      cll::init("NA"));
static cll::opt<bool> outOnlyP("outdeg",
                               cll::desc("Out degree only for priority"),
                               cll::init(false));
static cll::opt<Algo>
    algo("algo", cll::desc("Choose an algorithm:"),
         cll::values(
             clEnumValN(Algo::asyncB, "asyncB", "Asynchronous versoin..."),
             clEnumValN(Algo::asyncB_hset, "asyncB_hset",
                        "asyncB with a two-level hash uni-set scheduler"),
             clEnumValN(Algo::asyncB_mset, "asyncB_mset",
                        "asyncB with an item-marking uni-set scheduler"),
             clEnumValN(Algo::asyncB_oset, "asyncB_oset",
                        "asyncB with a two-level set uni-set scheduler"),
             clEnumValN(Algo::asyncB_prt, "asyncB_prt",
                        "Prioritized (degree biased residual) version..."),
             clEnumValN(Algo::asyncB_prt_hset, "asyncB_prt_hset",
                        "asyncB_prt with a two-level hash uni-set scheduler"),
             clEnumValN(Algo::asyncB_prt_mset, "asyncB_prt_mset",
                        "asyncB_prt with an item-marking uni-set scheduler"),
             clEnumValN(Algo::asyncB_prt_oset, "asyncB_prt_oset",
                        "asyncB_prt with a two-level set uni-set scheduler"),
             clEnumValEnd),
         cll::init(Algo::asyncB));

bool outOnly;

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
    float value           = n.getPageRank(0);
    sum += value;
  }

  Top top;

  std::ofstream myfile;
  if (dbg) {
    char filename[256];
    int tamp   = amp;
    float ttol = tolerance;
    sprintf(filename, "/scratch/01982/joyce/tmp/%s_t_%d_tol_%f_amp_%d",
            algo_name, numThreads, ttol, tamp);
    myfile.open(filename);
  }

  // std::cout<<"print PageRank\n";
  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src             = *ii;
    node_data_reference n = graph.getData(src);
    float value = n.getPageRank(0) / sum; // normalized PR (divide PR by sum)
    // float value = n.getPageRank(); // raw PR
    // std::cout<<value<<" ";
    if (dbg) {
      myfile << value << " ";
    }
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
  if (dbg) {
    myfile.close();
  }
  // std::cout<<"\nend of print\n";

  int rank = 1;
  std::cout << "Rank PageRank Id\n";
  for (typename Top::reverse_iterator ii = top.rbegin(), ei = top.rend();
       ii != ei; ++ii, ++rank) {
    std::cout << rank << ": " << ii->first.value << " " << ii->first.id << "\n";
  }
}

template <typename Graph>
struct LNodeSetMarker
    : public std::unary_function<typename Graph::GraphNode, bool*> {
  Graph& graph;
  LNodeSetMarker(Graph& g) : graph(g) {}

  bool* operator()(const typename Graph::GraphNode n) const {
    return &(graph.getData(n, galois::MethodFlag::UNPROTECTED).inSet);
  }
};

struct Async {
  struct LNode {
    PRTy value;
    bool inSet;
    void init() {
      value = 1.0 - alpha;
      inSet = false;
    }
    PRTy getPageRank(int x = 0) { return value; }
    friend std::ostream& operator<<(std::ostream& os, const LNode& n) {
      os << "{PR " << n.value << ", inSet " << n.inSet << "}";
      return os;
    }
  };

  typedef galois::graphs::LC_CSR_Graph<LNode, void>::with_numa_alloc<true>::type
      InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "Async"; }

  void readGraph(Graph& graph, std::string filename,
                 std::string transposeGraphName) {
    check_types<Graph, InnerGraph>();
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, filename, transposeGraphName);
    } else {
      std::cerr
          << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct Process {
    Graph& graph;
    PRTy tolerance;

    Process(Graph& g, PRTy t) : graph(g), tolerance(t) {}

    void operator()(const GNode& src, galois::UserContext<GNode>& ctx) const {
      LNode& sdata                = graph.getData(src);
      galois::MethodFlag lockflag = galois::MethodFlag::UNPROTECTED;

      PRTy pr   = computePageRankInOut(graph, src, 0, lockflag);
      PRTy diff = std::fabs(pr - sdata.value);
      if (diff >= tolerance) {
        sdata.value = pr;
        // for each out-going neighbors
        for (auto jj = graph.edge_begin(src, lockflag),
                  ej = graph.edge_end(src, lockflag);
             jj != ej; ++jj) {
          GNode dst    = graph.getEdgeDst(jj);
          LNode& ddata = graph.getData(dst, lockflag);
          ctx.push(dst);
        }
      }
    }
  };

  void operator()(Graph& graph, PRTy tolerance, PRTy amp) {
    typedef galois::worklists::PerSocketChunkFIFO<16> WL;
    typedef galois::worklists::PerSocketChunkTwoLevelHashFIFO<16> HSet;
    typedef galois::worklists::PerSocketChunkTwoLevelSetFIFO<16> OSet;
    typedef galois::worklists::PerSocketChunkMarkingSetFIFO<
        LNodeSetMarker<Graph>, 16>
        MSet;

    auto marker = LNodeSetMarker<Graph>(graph);

    if (algo == Algo::asyncB_hset)
      galois::for_each(graph, Process(graph, tolerance), galois::wl<HSet>());
    else if (algo == Algo::asyncB_mset)
      galois::for_each(graph, Process(graph, tolerance),
                       galois::wl<MSet>(marker));
    else if (algo == Algo::asyncB_oset)
      galois::for_each(graph, Process(graph, tolerance), galois::wl<OSet>());
    else
      galois::for_each(graph, Process(graph, tolerance), galois::wl<WL>());
  }

  void verify(Graph& graph, PRTy tolerance) { verifyInOut(graph, tolerance); }
};

struct AsyncNodePri {
  struct LNode {
    PRTy value;
    std::atomic<PRTy> residual;
    bool inSet;
    void init() {
      value    = 1.0 - alpha;
      residual = 0.0;
      inSet    = false;
    }
    PRTy getPageRank(int x = 0) { return value; }
    friend std::ostream& operator<<(std::ostream& os, const LNode& n) {
      os << "{PR " << n.value << ", residual " << n.residual << ", inSet "
         << n.inSet << "}";
      return os;
    }
  };

  typedef galois::graphs::LC_CSR_Graph<LNode, void>::with_numa_alloc<true>::type
      InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "AsyncNodePri"; }

  void readGraph(Graph& graph, std::string filename,
                 std::string transposeGraphName) {
    if (transposeGraphName.size()) {
      galois::graphs::readGraph(graph, filename, transposeGraphName);
    } else {
      std::cerr
          << "Need to pass precomputed graph through -graphTranspose option\n";
      abort();
    }
  }

  struct PRPri {
    Graph& graph;
    PRTy tolerance;
    PRPri(Graph& g, PRTy t) : graph(g), tolerance(t) {}
    int operator()(const GNode src, PRTy d) const {
      if (outOnly)
        d /= (1 + nout(graph, src, galois::MethodFlag::UNPROTECTED));
      else
        d /= ninout(graph, src, galois::MethodFlag::UNPROTECTED);
      d /= tolerance;
      if (d > 50)
        return -50;
      return -d; // d*amp; //std::max((int)floor(d*amp), 0);
    }
    int operator()(const GNode src) const {
      PRTy d = graph.getData(src, galois::MethodFlag::UNPROTECTED).residual;
      return operator()(src, d);
    }
  };

  struct Process {
    Graph& graph;
    PRTy tolerance;
    PRPri pri;

    Process(Graph& g, PRTy t, PRTy a) : graph(g), tolerance(t), pri(g, t) {}

    void operator()(const GNode src, galois::UserContext<GNode>& ctx) const {
      LNode& sdata = graph.getData(src);

      if (sdata.residual < tolerance)
        return;

      galois::MethodFlag lockflag = galois::MethodFlag::UNPROTECTED;

      PRTy oldResidual = sdata.residual.exchange(0.0);
      PRTy pr          = computePageRankInOut(graph, src, 0, lockflag);
      PRTy diff        = std::fabs(pr - sdata.value);
      sdata.value      = pr;
      int src_nout     = nout(graph, src, lockflag);
      PRTy delta       = diff * alpha / src_nout;
      // for each out-going neighbors
      for (auto jj = graph.edge_begin(src, lockflag),
                ej = graph.edge_end(src, lockflag);
           jj != ej; ++jj) {
        GNode dst    = graph.getEdgeDst(jj);
        LNode& ddata = graph.getData(dst, lockflag);
        PRTy old     = atomicAdd(ddata.residual, delta);
        // if the residual is greater than tolerance
        if (old + delta >= tolerance) {
          // std::cerr << pri(dst, old+delta) << " ";
          ctx.push(dst);
        }
      }
    }
  };

  void operator()(Graph& graph, PRTy tolerance, PRTy amp) {
    initResidual(graph);
    using namespace galois::worklists;
    typedef PerSocketChunkFIFO<32> WL;
    typedef OrderedByIntegerMetric<PRPri, WL>::with_block_period<8>::type OBIM;
    typedef detail::MarkingWorkSetMaster<GNode, LNodeSetMarker<Graph>, OBIM>
        ObimMSet;
    typedef detail::WorkSetMaster<GNode, OBIM,
                                  galois::ThreadSafeTwoLevelSet<GNode>>
        ObimOSet;
    typedef detail::WorkSetMaster<GNode, OBIM,
                                  galois::ThreadSafeTwoLevelHash<GNode>>
        ObimHSet;

    galois::InsertBag<GNode> bag;
    PRPri pri(graph, tolerance);
    auto marker = LNodeSetMarker<Graph>(graph);
    // galois::do_all(graph, [&graph, &bag, &pri] (const GNode& node) {
    //     bag.push(std::make_pair(node, pri(node)));
    //   });
    // galois::for_each(bag, Process(graph, tolerance, amp),
    // galois::wl<OBIM>());

    if (algo == Algo::asyncB_prt_mset)
      galois::for_each(graph.begin(), graph.end(),
                       Process(graph, tolerance, amp),
                       galois::wl<ObimMSet>(marker, dummy, pri));
    else if (algo == Algo::asyncB_prt_oset)
      galois::for_each(graph.begin(), graph.end(),
                       Process(graph, tolerance, amp),
                       galois::wl<ObimOSet>(dummy, pri));
    else if (algo == Algo::asyncB_prt_hset)
      galois::for_each(graph.begin(), graph.end(),
                       Process(graph, tolerance, amp),
                       galois::wl<ObimHSet>(dummy, pri));
    else
      galois::for_each(graph.begin(), graph.end(),
                       Process(graph, tolerance, amp), galois::wl<OBIM>(pri));
  }

  void verify(Graph& graph, PRTy tolerance) { verifyInOut(graph, tolerance); }
};

template <typename Algo>
void run() {
  typedef typename Algo::Graph Graph;

  Algo algo;
  Graph graph;

  algo.readGraph(graph, filename, transposeGraphName);

  galois::preAlloc(numThreads +
                   (2 * graph.size() * sizeof(typename Graph::node_data_type)) /
                       galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  galois::StatTimer T;
  auto eamp = -amp; /// tolerance;
  std::cout << "Running " << algo.name() << " version\n";
  std::cout << "tolerance: " << tolerance << "\n";
  std::cout << "effective amp: " << eamp << "\n";
  T.start();
  galois::do_all(graph, [&graph](typename Graph::GraphNode n) {
    graph.getData(n).init();
  });
  algo(graph, tolerance, eamp);
  T.stop();

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    algo.verify(graph, tolerance);
    printTop(graph, 10, algo.name().c_str(), numThreads);
  }
}

int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);
  galois::StatManager statManager;

  outOnly = outOnlyP;

  galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
  case Algo::asyncB:
    run<Async>();
    break;
  case Algo::asyncB_hset:
    run<Async>();
    break;
  case Algo::asyncB_mset:
    run<Async>();
    break;
  case Algo::asyncB_oset:
    run<Async>();
    break;
  case Algo::asyncB_prt:
    run<AsyncNodePri>();
    break;
  case Algo::asyncB_prt_hset:
    run<AsyncNodePri>();
    break;
  case Algo::asyncB_prt_mset:
    run<AsyncNodePri>();
    break;
  case Algo::asyncB_prt_oset:
    run<AsyncNodePri>();
    break;
  default:
    std::cerr << "Unknown algorithm\n";
    abort();
  }
  T.stop();

  return 0;
}
