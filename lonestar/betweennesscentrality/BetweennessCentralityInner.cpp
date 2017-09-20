/** Betweenness Centrality -*- C++ -*-
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
 * Betweenness centrality. Implementation based on Ligra
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Timer.h"
#include "Galois/Graphs/LCGraph.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/Graphs/GraphNodeBag.h"
#include "HybridBFS.h"

#include <atomic>
#include <string>
#include <deque>
#include <iostream>
#include <iomanip>

static const char* name = "Betweenness Centrality";
static const char* desc = 0;
static const char* url = 0;

enum Algo {
  async,
  leveled
};

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<std::string> transposeGraphName("graphTranspose", cll::desc("Transpose of input graph"));
static cll::opt<bool> symmetricGraph("symmetricGraph", cll::desc("Input graph is symmetric"));
static cll::opt<unsigned int> startNode("startNode", cll::desc("Node to start search from"), cll::init(0));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumValN(Algo::async, "async", "Async Algorithm"),
      clEnumValN(Algo::leveled, "leveled", "Leveled Algorithm"),
      clEnumValEnd), cll::init(Algo::async));

template<typename Algo>
void initialize(Algo& algo,
    typename Algo::Graph& graph,
    typename Algo::Graph::GraphNode& source) {

  algo.readGraph(graph);
  std::cout << "Read " << graph.size() << " nodes\n";

  if (startNode >= graph.size()) {
    std::cerr << "failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }
  
  typename Algo::Graph::iterator it = graph.begin();
  std::advance(it, startNode);
  source = *it;
}

template<typename Graph>
void readInOutGraph(Graph& graph) {
  using namespace galois::Graph;
  if (symmetricGraph) {
    galois::Graph::readGraph(graph, filename);
  } else if (transposeGraphName.size()) {
    galois::Graph::readGraph(graph, filename, transposeGraphName);
  } else {
    GALOIS_DIE("Graph type not supported");
  }
}

static const int ChunkSize = 128;
static const int bfsChunkSize = 32+64;

struct AsyncAlgo {
  struct SNode {
    float numPaths;
    float dependencies;
    int dist;
    SNode() :numPaths(-std::numeric_limits<float>::max()), dependencies(-std::numeric_limits<float>::max()), dist(std::numeric_limits<int>::max()) { }
  };

  typedef galois::Graph::LC_CSR_Graph<SNode,void>
    ::with_no_lockable<true>::type 
    ::with_numa_alloc<true>::type InnerGraph;
  typedef galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
//typedef galois::Graph::LC_CSR_Graph<SNode, void> Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "async"; }

  void readGraph(Graph& graph) { 
    readInOutGraph(graph);
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) const {
      SNode& data = g.getData(n, galois::MethodFlag::UNPROTECTED);
      data.numPaths = -std::numeric_limits<float>::max();
      data.dependencies = -std::numeric_limits<float>::max();
      data.dist = std::numeric_limits<int>::max();
    }
  };

  struct BFS {
    typedef int tt_does_not_need_aborts;
    typedef std::pair<GNode, int> WorkItem;
    
    struct Indexer: public std::unary_function<WorkItem,int> {
      int operator()(const WorkItem& val) const {
        return val.second;
      }
    };

    typedef galois::WorkList::OrderedByIntegerMetric<Indexer,galois::WorkList::dChunkedFIFO<bfsChunkSize> > OBIM;

    Graph& g;
    BFS(Graph& g) :g(g) {}

    void operator()(WorkItem& item, galois::UserContext<WorkItem>& ctx) const {
      GNode n = item.first;
      int newDist = item.second;
      if (newDist > g.getData(n).dist + 1)
        return;
      
      for (auto ii : g.edges(n, galois::MethodFlag::UNPROTECTED)) {
        GNode dst = g.getEdgeDst(ii);
        SNode& ddata = g.getData(dst, galois::MethodFlag::UNPROTECTED);

        int oldDist;
        while (true) {
          oldDist = ddata.dist;
          if (oldDist <= newDist)
            break;
          if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
            ctx.push(WorkItem(dst, newDist + 1));
            break;
          }
        }
      }
    }
  };

  struct CountPaths {
    typedef int tt_does_not_need_aborts;

    struct Indexer: public std::unary_function<GNode,int> {
      static Graph* g;
      int operator()(const GNode& val) const {
        // //use out edges as that signifies how many people will wait on this node
        // auto ii = g->edge_begin(val, galois::MethodFlag::UNPROTECTED);
        // auto ee = g->edge_end(val, galois::MethodFlag::UNPROTECTED);
        // bool big = galois::safe_advance(ii, ee, 10) != ee;
        // return 2 * g->getData(val, galois::MethodFlag::UNPROTECTED).dist + (big ? 0 : 1);
        return g->getData(val, galois::MethodFlag::UNPROTECTED).dist;
      }
    };

    typedef galois::WorkList::OrderedByIntegerMetric<Indexer,galois::WorkList::dChunkedFIFO<ChunkSize> > OBIM;

    Graph& g;
    CountPaths(Graph& g) :g(g) { Indexer::g = &g; }

    void operator()(GNode& n, galois::UserContext<GNode>& ctx) const {
      SNode& sdata = g.getData(n, galois::MethodFlag::UNPROTECTED);
      while (sdata.numPaths == -std::numeric_limits<float>::max()) {
        unsigned long np = 0;
        bool allready = true;
        for (Graph::in_edge_iterator ii = g.in_edge_begin(n, galois::MethodFlag::UNPROTECTED),
               ee = g.in_edge_end(n, galois::MethodFlag::UNPROTECTED); ii != ee; ++ii) {
          GNode dst = g.getInEdgeDst(ii);
          SNode& ddata = g.getData(dst, galois::MethodFlag::UNPROTECTED);
          if (ddata.dist + 1 == sdata.dist) {
            if (ddata.numPaths != -std::numeric_limits<float>::max()) {
              np += ddata.numPaths;
            } else {
              allready = false;
              // ctx.push(n);
              // return;
            }
          }
        }
        if (allready)
          sdata.numPaths = np;
      }
    }
  };

  struct ComputeDep {
    typedef int tt_does_not_need_aborts;
    
    struct Indexer: public std::unary_function<GNode,int> {
      static Graph* g;
      int operator()(const GNode& val) const {
        //use in edges as that signifies how many people will wait on this node
        // auto ii = g->in_edge_begin(val, galois::MethodFlag::UNPROTECTED);
        // auto ee = g->in_edge_end(val, galois::MethodFlag::UNPROTECTED);
        // bool big = galois::safe_advance(ii, ee, 10) != ee;
        // return std::numeric_limits<int>::max() - 2 * g->getData(val, galois::MethodFlag::UNPROTECTED).dist + (big ? 0 : 1);
        return std::numeric_limits<int>::max() - g->getData(val, galois::MethodFlag::UNPROTECTED).dist;
      }
    };

    typedef galois::WorkList::OrderedByIntegerMetric<Indexer,galois::WorkList::dChunkedFIFO<ChunkSize> > OBIM;

    Graph& g;
    ComputeDep(Graph& g) :g(g) { Indexer::g = &g; }

    void operator()(GNode& n, galois::UserContext<GNode>& ctx) const {
      SNode& sdata = g.getData(n, galois::MethodFlag::UNPROTECTED);
      while (sdata.dependencies == -std::numeric_limits<float>::max()) {
        float newDep = 0.0;
        bool allready = true;
        for (auto ii : g.edges(n, galois::MethodFlag::UNPROTECTED)) {
          GNode dst = g.getEdgeDst(ii);
          SNode& ddata = g.getData(dst, galois::MethodFlag::UNPROTECTED);
          if (ddata.dist == sdata.dist + 1) {
            if (ddata.dependencies != -std::numeric_limits<float>::max()) {
              newDep += ((float)sdata.numPaths / (float)ddata.numPaths) * (1 + ddata.dependencies);
            } else {
              allready = false;
              // ctx.push(n);
              // return;
            }
          }
        }
        if (allready)
          sdata.dependencies = newDep;
      }
    }
  };

  void operator()(Graph& graph, GNode source) {
    galois::StatTimer Tinit("InitTime"), Tlevel("LevelTime"), Tbfs("BFSTime"), Tcount("CountTime"), Tdep("DepTime");
    Tinit.start();
    galois::do_all_local(graph, Initialize(graph), galois::loopname("INIT"));
    Tinit.stop();
    std::cout << "INIT DONE " << Tinit.get() << "\n";
    Tbfs.start();
    graph.getData(source).dist = 0;
    //galois::for_each(BFS::WorkItem(source, 1), BFS(graph), galois::loopname("BFS"), galois::wl<BFS::OBIM>());
    HybridBFS<SNode, int> H;
    H(graph,source);
    Tbfs.stop();
    std::cout << "BFS DONE " << Tbfs.get() << "\n";
    Tcount.start();
    graph.getData(source).numPaths = 1;
    galois::for_each_local(graph, CountPaths(graph), galois::loopname("COUNT"), galois::wl<CountPaths::OBIM>());
    Tcount.stop();
    std::cout << "COUNT DONE " << Tcount.get() << "\n";
    Tdep.start();
    graph.getData(source).dependencies = 0.0;
    galois::for_each(graph.begin(), graph.end(), ComputeDep(graph), galois::loopname("DEP"), galois::wl<ComputeDep::OBIM>());
    Tdep.stop();
    std::cout << "DEP DONE " << Tdep.get() << "\n";
  }
};

AsyncAlgo::Graph* AsyncAlgo::CountPaths::Indexer::g;
AsyncAlgo::Graph* AsyncAlgo::ComputeDep::Indexer::g;

struct LeveledAlgo {
  struct SNode {
    std::atomic<unsigned long> numPaths;
    float dependencies;
    std::atomic<int> dist;
    SNode() :numPaths(~0UL), dependencies(-std::numeric_limits<float>::max()), dist(std::numeric_limits<int>::max()) { }
  };

  typedef galois::Graph::LC_CSR_Graph<SNode,void>
    ::with_no_lockable<true>::type 
    ::with_numa_alloc<true>::type InnerGraph;
  typedef galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
//typedef galois::Graph::LC_CSR_Graph<SNode, void> Graph;
  typedef Graph::GraphNode GNode;
  typedef galois::InsertBag<GNode> Bag;

  std::string name() const { return "Leveled"; }

  void readGraph(Graph& graph) { 
    readInOutGraph(graph);
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(Graph::GraphNode n) const {
      SNode& data = g.getData(n, galois::MethodFlag::UNPROTECTED);
      data.numPaths = 0;
      data.dependencies = 0.0; //std::numeric_limits<float>::lowest();
      data.dist = std::numeric_limits<int>::max();
    }
  };

  //push based
  template<bool doCount = true>
  struct BFS {
    typedef int tt_does_not_need_aborts;

    Graph& g;
    Bag& b;
    BFS(Graph& g, Bag& b) :g(g), b(b) {}

    void operator()(GNode& n) const {
      auto& sdata = g.getData(n, galois::MethodFlag::UNPROTECTED);
      for (auto ii : g.edges(n, galois::MethodFlag::UNPROTECTED)) {
        GNode dst = g.getEdgeDst(ii);
        SNode& ddata = g.getData(dst, galois::MethodFlag::UNPROTECTED);
        if (ddata.dist.load(std::memory_order_relaxed) == std::numeric_limits<int>::max()) {
          if (std::numeric_limits<int>::max() == ddata.dist.exchange(sdata.dist + 1))
            b.push_back(dst);
          if (doCount)
            ddata.numPaths = ddata.numPaths + sdata.numPaths;
        } else if (ddata.dist == sdata.dist + 1) {
          if (doCount)
            ddata.numPaths = ddata.numPaths + sdata.numPaths;
        }
      }
      // for (Graph::in_edge_iterator ii = g.in_edge_begin(n, galois::MethodFlag::UNPROTECTED),
      //        ee = g.in_edge_end(n, galois::MethodFlag::UNPROTECTED); ii != ee; ++ii) {
      //   GNode dst = g.getInEdgeDst(ii);
      //   SNode& ddata = g.getData(dst, galois::MethodFlag::UNPROTECTED);
      //   if (ddata.dist + 1 == sdata.dist)
      //     sdata.numPaths += ddata.numPaths;
      // }
    }
  };

  //pull based
  struct Counter {
    typedef int tt_does_not_need_aborts;

    Graph& g;
    Counter(Graph& g) :g(g) {}

    void operator()(GNode& n) const {
      auto& sdata = g.getData(n, galois::MethodFlag::UNPROTECTED);
      unsigned long np = 0;
      for (Graph::in_edge_iterator ii = g.in_edge_begin(n, galois::MethodFlag::UNPROTECTED),
             ee = g.in_edge_end(n, galois::MethodFlag::UNPROTECTED); ii != ee; ++ii) {
        GNode dst = g.getInEdgeDst(ii);
        SNode& ddata = g.getData(dst, galois::MethodFlag::UNPROTECTED);
        if (ddata.dist + 1 == sdata.dist)
          np += ddata.numPaths;
      }
      sdata.numPaths = sdata.numPaths + np;
    }
  };

  //pull based
  struct ComputeDep {
    Graph& g;
    ComputeDep(Graph& g) :g(g) {}

    void operator()(GNode& n) const {
      SNode& sdata = g.getData(n, galois::MethodFlag::UNPROTECTED);
      for (auto ii : g.edges(n, galois::MethodFlag::UNPROTECTED)) {
        GNode dst = g.getEdgeDst(ii);
        SNode& ddata = g.getData(dst, galois::MethodFlag::UNPROTECTED);
        if (ddata.dist == sdata.dist + 1)
          sdata.dependencies += ((float)sdata.numPaths / (float)ddata.numPaths) * (1 + ddata.dependencies);
      }
    }
  };

  void operator()(Graph& graph, GNode source) {
    galois::StatTimer Tinit("InitTime");
    galois::StatTimer Tlevel("LevelTime");
    galois::StatTimer Tbfs("BFSTime");
    galois::StatTimer Tcount("CountTime");
    galois::StatTimer Tdep("DepTime");
    Tinit.start();
    galois::do_all_local(graph, Initialize(graph), galois::loopname("INIT"));
    Tinit.stop();
    std::cout << "INIT DONE " << Tinit.get() << "\n";

    Tbfs.start();
    std::deque<Bag*> levels;
    levels.push_back(new Bag());
    levels[0]->push_back(source);
    graph.getData(source).dist = 0;
    graph.getData(source).numPaths = 1;
    while (!levels.back()->empty()) {
      Bag* b = levels.back();
      levels.push_back(new Bag());
      galois::do_all_local(*b, BFS<>(graph, *levels.back()), galois::loopname("BFS"), galois::do_all_steal<true>());
      //galois::do_all_local(*levels.back(), Counter(graph), "COUNTER", true);
    }
    delete levels.back();
    levels.pop_back();
    Tbfs.stop();
    std::cout << "BFS DONE " << Tbfs.get() << " with " << levels.size() << " levels\n";

    Tdep.start();
    for (int i = levels.size() - 1; i > 0; --i)
      galois::do_all_local(*levels[i-1], ComputeDep(graph), galois::loopname("DEPS"), galois::do_all_steal<true>());
    Tdep.stop();
    std::cout << "DEP DONE " << Tdep.get() << "\n";
    while (!levels.empty()) {
      delete levels.back();
      levels.pop_back();
    }     
  }
};


template<typename Algo>
void run() {
  typedef typename Algo::Graph Graph;
  typedef typename Graph::GraphNode GNode;

  Algo algo;
  Graph graph;
  GNode source;

  initialize(algo, graph, source);

  galois::reportPageAlloc("MeminfoPre");
  galois::preAlloc(numThreads + (3*graph.size() * sizeof(typename Graph::node_data_type)) / galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoMid");

  galois::StatTimer T;
  std::cout << "Running " << algo.name() << " version\n";
  T.start();
  algo(graph, source);
  T.stop();
  
  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    int count = 0;
    for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei && count < 20; ++ii, ++count) {
      std::cout << count << ": "
        << std::setiosflags(std::ios::fixed) << std::setprecision(6) 
          << graph.getData(*ii).dependencies
        << " " << graph.getData(*ii).numPaths 
        << " " << graph.getData(*ii).dist
        << "\n";
    }
    //count = 0;
    // for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii, ++count)
    //   std::cout << ((count % 128 == 0) ? "\n" : " ") << graph.getData(*ii).numPaths;
    //std::cout << "\n";
  }
}

int main(int argc, char **argv) {
  galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
    case Algo::async: run<AsyncAlgo>();   break;
    case Algo::leveled: run<LeveledAlgo>(); break;
  }
  T.stop();

  return 0;
}
