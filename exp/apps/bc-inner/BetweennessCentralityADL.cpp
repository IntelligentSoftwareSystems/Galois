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
#include "Galois/config.h"
#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#ifdef GALOIS_USE_EXP
#include <boost/mpl/if.hpp>
#include "Galois/Graph/OCGraph.h"
#include "Galois/Graph/GraphNodeBag.h"
#include "Galois/DomainSpecificExecutors.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include GALOIS_CXX11_STD_HEADER(atomic)
#include <string>
#include <deque>
#include <iostream>
#include <iomanip>

static const char* name = "Betweenness Centrality";
static const char* desc = 0;
static const char* url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<std::string> transposeGraphName("graphTranspose", cll::desc("Transpose of input graph"));
static cll::opt<bool> symmetricGraph("symmetricGraph", cll::desc("Input graph is symmetric"));
static cll::opt<unsigned int> startNode("startNode", cll::desc("Node to start search from"), cll::init(0));

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
  using namespace Galois::Graph;
  if (symmetricGraph) {
    Galois::Graph::readGraph(graph, filename);
  } else if (transposeGraphName.size()) {
    Galois::Graph::readGraph(graph, filename, transposeGraphName);
  } else {
    GALOIS_DIE("Graph type not supported");
  }
}

struct ADLAlgo: public Galois::LigraGraphChi::ChooseExecutor<false> {

  struct SNode {
    std::atomic<int> numPaths;
    std::atomic<int> numPathsProp;
    std::atomic<double> dependencies;
    int dist;
    std::atomic<bool> visited;
    SNode() { }
  };

  typedef typename Galois::Graph::LC_CSR_Graph<SNode,void>
    ::with_no_lockable<true>::type 
    ::with_numa_alloc<true>::type InnerGraph;

  typedef typename boost::mpl::if_c<false,
          Galois::Graph::OCImmutableEdgeGraph<SNode,void>,
          Galois::Graph::LC_InOut_Graph<InnerGraph> >::type
          Graph;
  typedef typename Graph::GraphNode GNode;
  typedef Galois::GraphNodeBag<1024*4> Bag;

  std::string name() const { return "ADL"; }

  void readGraph(Graph& graph) { 
    readInOutGraph(graph);
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(typename Graph::GraphNode n) {
      SNode& data = g.getData(n, Galois::MethodFlag::NONE);
      data.numPaths = 0;
      data.numPathsProp = 0;
      data.dependencies = 0.0;
      data.visited = false;
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

    typedef Galois::WorkList::OrderedByIntegerMetric<Indexer,Galois::WorkList::dChunkedFIFO<64> > OBIM;

    Graph& g;
    BFS(Graph& g) :g(g) {}

    void operator()(WorkItem& item, Galois::UserContext<WorkItem>& ctx) const {
      GNode n = item.first;
      int newDist = item.second;
      
      for (Graph::edge_iterator ii = g.edge_begin(n, Galois::MethodFlag::NONE),
            ei = g.edge_end(n, Galois::MethodFlag::NONE); ii != ei; ++ii) {
        GNode dst = g.getEdgeDst(ii);
        SNode& ddata = g.getData(dst, Galois::MethodFlag::NONE);

        while (true) {
          int oldDist = ddata.dist;
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
        return g->getData(val, Galois::MethodFlag::NONE).dist;
      }
    };

    typedef Galois::WorkList::OrderedByIntegerMetric<Indexer,Galois::WorkList::dChunkedFIFO<64> > OBIM;

    Graph& g;
    CountPaths(Graph& g) :g(g) { Indexer::g = &g; }

    void operator()(GNode& n, Galois::UserContext<GNode>& ctx) const {
      SNode& sdata = g.getData(n, Galois::MethodFlag::NONE);
      int toProp = 0;
      int oldPaths, oldPathsProp;
      while ((oldPaths = sdata.numPaths) > (oldPathsProp = sdata.numPathsProp)) {
        if (sdata.numPathsProp.compare_exchange_weak(oldPathsProp, oldPaths)) {
          toProp = oldPaths - oldPathsProp;
          break;
        }
      }
      if (toProp) {
        for (Graph::edge_iterator ii = g.edge_begin(n, Galois::MethodFlag::NONE),
               ei = g.edge_end(n, Galois::MethodFlag::NONE); ii != ei; ++ii) {
          GNode dst = g.getEdgeDst(ii);
          SNode& ddata = g.getData(dst, Galois::MethodFlag::NONE);
          if (ddata.dist == sdata.dist + 1) {
            ddata.numPaths += toProp;
            ctx.push(dst);
          }
        }
      }
    }
  };

  struct ComputeDep {
    typedef int tt_does_not_need_aborts;
    
    struct Indexer: public std::unary_function<GNode,int> {
      static Graph* g;
      int operator()(const GNode& val) const {
        return std::numeric_limits<int>::max() - g->getData(val, Galois::MethodFlag::NONE).dist;
      }
    };

    typedef Galois::WorkList::OrderedByIntegerMetric<Indexer,Galois::WorkList::dChunkedFIFO<64> > OBIM;

    Graph& g;
    ComputeDep(Graph& g) :g(g) { Indexer::g = &g; }

    void operator()(GNode& n, Galois::UserContext<GNode>& ctx) const {
      SNode& sdata = g.getData(n);
      if (sdata.visited)
        return;
      bool allReady = true;
      double newDep = 0.0;
      for (Graph::edge_iterator ii = g.edge_begin(n, Galois::MethodFlag::NONE),
             ei = g.edge_end(n, Galois::MethodFlag::NONE); ii != ei; ++ii) {
        GNode dst = g.getEdgeDst(ii);
        SNode& ddata = g.getData(dst, Galois::MethodFlag::NONE);
        if (ddata.dist == sdata.dist + 1) {
          if (ddata.visited)
            newDep += ((double)sdata.numPaths / (double)ddata.numPaths) * (1 + ddata.dependencies);
          else
            allReady = false;
        }
      }
      if (allReady) {
        sdata.dependencies = newDep;
        sdata.visited = true;
        for (Graph::edge_iterator ii = g.edge_begin(n, Galois::MethodFlag::NONE),
               ei = g.edge_end(n, Galois::MethodFlag::NONE); ii != ei; ++ii) {
          GNode dst = g.getEdgeDst(ii);
          SNode& ddata = g.getData(dst, Galois::MethodFlag::NONE);
          if (ddata.dist + 1 == sdata.dist)
            ctx.push(dst);
        }
      }
    }
  };

  void operator()(Graph& graph, GNode source) {
    Galois::StatTimer Tinit("InitTime"), Tbfs("BFSTime"), Tcount("CountTime"), Tdep("DepTime");
    Tinit.start();
    Galois::do_all_local(graph, Initialize(graph));
    Tinit.stop();
    Tbfs.start();
    graph.getData(source).dist = 0;
    Galois::for_each<BFS::OBIM>(BFS::WorkItem(source, 1), BFS(graph));
    Tbfs.stop();
    Tcount.start();
    CountPaths::Indexer::g = &graph;
    graph.getData(source).numPaths = 1;
    Galois::for_each<CountPaths::OBIM>(source, CountPaths(graph));
    Tcount.stop();
    Tdep.start();
    ComputeDep::Indexer::g = &graph;
    Galois::for_each<ComputeDep::OBIM>(graph.begin(), graph.end(), ComputeDep(graph));
    Tdep.stop();
  }
};

ADLAlgo::Graph* ADLAlgo::CountPaths::Indexer::g;
ADLAlgo::Graph* ADLAlgo::ComputeDep::Indexer::g;

void run() {
  typedef typename ADLAlgo::Graph Graph;
  typedef typename Graph::GraphNode GNode;

  ADLAlgo algo;
  Graph graph;
  GNode source;

  initialize(algo, graph, source);

  Galois::preAlloc(numThreads + (3*graph.size() * sizeof(typename Graph::node_data_type)) / Galois::Runtime::MM::pageSize);
  Galois::reportPageAlloc("MeminfoPre");

  Galois::StatTimer T;
  std::cout << "Running " << algo.name() << " version\n";
  T.start();
  algo(graph, source);
  T.stop();
  
  Galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    int count = 0;
    for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei && count < 20; ++ii, ++count) {
      std::cout << count << ": "
        << std::setiosflags(std::ios::fixed) << std::setprecision(6) << graph.getData(*ii).dependencies
                << " " << graph.getData(*ii).numPaths
        << "\n";
    }
  }
}

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  Galois::StatTimer T("TotalTime");
  T.start();
  run();
  T.stop();

  return 0;
}
