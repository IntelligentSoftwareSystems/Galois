/** Betweenness Centrality -*- C++ -*-
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
 * @section Description
 *
 * Betweenness centrality. Implementation based on Ligra.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include "HybridBFS.h"

#include <iomanip>

static const char* name = "Betweenness Centrality";
static const char* desc = 0;
static const char* url = 0;

enum Algo {
  async,
  leveled
};

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, 
                                 cll::desc("<input graph>"), cll::Required);
static cll::opt<std::string> transposeGraphName("graphTranspose", 
                                 cll::desc("Transpose of input graph"));
static cll::opt<bool> symmetricGraph("symmetricGraph", 
                          cll::desc("Input graph is symmetric"));
static cll::opt<unsigned int> startNode("startNode", 
                                  cll::desc("Node to start search from"), 
                                  cll::init(0));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumValN(Algo::async, "async", "Async Algorithm"),
      clEnumValN(Algo::leveled, "leveled", "Leveled Algorithm"),
      clEnumValEnd), 
    cll::init(Algo::async));

template<typename Algo>
void initialize(Algo& algo, typename Algo::Graph& graph, 
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
  using namespace galois::graphs;
  if (symmetricGraph) {
    galois::graphs::readGraph(graph, filename);
  } else if (transposeGraphName.size()) {
    galois::graphs::readGraph(graph, filename, transposeGraphName);
  } else {
    GALOIS_DIE("Graph type not supported");
  }
}

static const int ChunkSize = 128;
static const int bfsChunkSize = 64;

struct AsyncAlgo {
  struct SNode {
    float numPaths;
    float dependencies;
    int dist;
    SNode() 
      : numPaths(-std::numeric_limits<float>::max()), 
        dependencies(-std::numeric_limits<float>::max()), 
        dist(std::numeric_limits<int>::max()) { }
  };

  typedef galois::graphs::LC_CSR_Graph<SNode,void>
    ::with_no_lockable<true>::type 
    ::with_numa_alloc<true>::type InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  std::string name() const { return "async"; }

  void readGraph(Graph& graph) { 
    readInOutGraph(graph);
  }

  void Initialize(Graph& graph) {
    galois::do_all(galois::iterate(graph),
        [&] (GNode n) {
          SNode& data = graph.getData(n, galois::MethodFlag::UNPROTECTED);
          data.numPaths = -std::numeric_limits<float>::max();
          data.dependencies = -std::numeric_limits<float>::max();
          data.dist = std::numeric_limits<int>::max();
        },
        galois::loopname("Initialize"));
  }

  void BFS(Graph& graph, GNode source) {
    using WorkItem =  std::pair<GNode, int>;

    auto indexer = [] (const WorkItem& val) { return val.second; };

    using OBIM = galois::worklists::OrderedByIntegerMetric<decltype(indexer)
      , galois::worklists::dChunkedFIFO<bfsChunkSize> >; 

    galois::for_each(
        galois::iterate({ WorkItem(source, 0) }),
        [&] (const WorkItem& item, auto& ctx) {
          GNode n = item.first;
          int newDist = item.second;
          if (newDist > graph.getData(n).dist + 1)
            return;
          
          for (auto ii : graph.edges(n, galois::MethodFlag::UNPROTECTED)) {
            GNode dst = graph.getEdgeDst(ii);
            SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

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
        },
        galois::no_conflicts(),
        galois::wl<OBIM>(indexer),
        galois::loopname("BFS"));
  }

  void CountPaths(Graph& graph) {
    auto indexer = [&] (const GNode& n) {
      return graph.getData(n, galois::MethodFlag::UNPROTECTED).dist;
    };

    using OBIM = galois::worklists::OrderedByIntegerMetric<decltype(indexer)
      , galois::worklists::dChunkedFIFO<ChunkSize> > ;


    galois::for_each(galois::iterate(graph),
        [&] (GNode n, auto& ctx) {
          SNode& sdata = graph.getData(n, galois::MethodFlag::UNPROTECTED);
          while (sdata.numPaths == -std::numeric_limits<float>::max()) {
            unsigned long np = 0;
            bool allready = true;
            for (Graph::in_edge_iterator ii = graph.in_edge_begin(n, galois::MethodFlag::UNPROTECTED),
                   ee = graph.in_edge_end(n, galois::MethodFlag::UNPROTECTED); ii != ee; ++ii) {
              GNode dst = graph.getInEdgeDst(ii);
              SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
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
          } // end while
        },
        galois::no_conflicts(),
        galois::no_pushes(),
        galois::wl<OBIM>(indexer),
        galois::loopname("CountPaths"));

  }

  void ComputeDep(Graph& graph) {

    auto indexer = [&] (const GNode& n) {
      return std::numeric_limits<int>::max() - graph.getData(n, galois::MethodFlag::UNPROTECTED).dist;
    };

    using OBIM = galois::worklists::OrderedByIntegerMetric<decltype(indexer)
      , galois::worklists::dChunkedFIFO<ChunkSize> > ;

    galois::for_each(galois::iterate(graph),
        [&] (GNode n, auto& ctx) {
          SNode& sdata = graph.getData(n, galois::MethodFlag::UNPROTECTED);
          while (sdata.dependencies == -std::numeric_limits<float>::max()) {
            float newDep = 0.0;
            bool allready = true;
            for (auto ii : graph.edges(n, galois::MethodFlag::UNPROTECTED)) {
              GNode dst = graph.getEdgeDst(ii);
              SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
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
          } // end while

        },
        galois::no_conflicts(),
        galois::no_pushes(),
        galois::wl<OBIM>(indexer),
        galois::loopname("ComputeDep"));

  }


  void operator()(Graph& graph, GNode source) {

    Initialize(graph);

    galois::StatTimer Tbfs("Tbfs");
    Tbfs.start();

    graph.getData(source).dist = 0;
    HybridBFS<SNode, int> H;
    H(graph, source);

    Tbfs.stop();

    graph.getData(source).numPaths = 1;
    CountPaths(graph);

    graph.getData(source).dependencies = 0.0;
    ComputeDep(graph);
  }
};

struct LeveledAlgo {
  struct SNode {
    std::atomic<unsigned long> numPaths;
    float dependencies;
    std::atomic<int> dist;
    SNode() :numPaths(~0UL), dependencies(-std::numeric_limits<float>::max()), dist(std::numeric_limits<int>::max()) { }
  };

  typedef galois::graphs::LC_CSR_Graph<SNode,void>
    ::with_no_lockable<true>::type 
    ::with_numa_alloc<true>::type InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
//typedef galois::graphs::LC_CSR_Graph<SNode, void> Graph;
  typedef Graph::GraphNode GNode;
  typedef galois::InsertBag<GNode> Bag;

  std::string name() const { return "Leveled"; }

  void readGraph(Graph& graph) { 
    readInOutGraph(graph);
  }

  void Initialize(Graph& graph) {
    galois::do_all(galois::iterate(graph),
        [&] (GNode n) {
          SNode& data = graph.getData(n, galois::MethodFlag::UNPROTECTED);
          data.numPaths = 0;
          data.dependencies = 0.0; //std::numeric_limits<float>::lowest();
          data.dist = std::numeric_limits<int>::max();
        },
        galois::loopname("Initialize"));
  }

  template <typename Deq>
  void BFS(Graph& graph, GNode source, Deq& levels) {

    constexpr static const bool doCount = true;

    galois::StatTimer Tbfs("BFSTime");
    Tbfs.start();

    levels.push_back(new Bag());
    levels[0]->push_back(source);
    graph.getData(source).dist = 0;
    graph.getData(source).numPaths = 1;

    while (!levels.back()->empty()) {
      Bag* b = levels.back();
      levels.push_back(new Bag());

      galois::do_all(galois::iterate(*b), 
          [&] (GNode n) {
            auto& sdata = graph.getData(n, galois::MethodFlag::UNPROTECTED);
            for (auto ii : graph.edges(n, galois::MethodFlag::UNPROTECTED)) {
              GNode dst = graph.getEdgeDst(ii);
              SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
              if (ddata.dist.load(std::memory_order_relaxed) == std::numeric_limits<int>::max()) {
                if (std::numeric_limits<int>::max() == ddata.dist.exchange(sdata.dist + 1))
                  b->push_back(dst);
                if (doCount)
                  ddata.numPaths = ddata.numPaths + sdata.numPaths;
              } else if (ddata.dist == sdata.dist + 1) {
                if (doCount)
                  ddata.numPaths = ddata.numPaths + sdata.numPaths;
              }
            }
            // for (Graph::in_edge_iterator ii = graph.in_edge_begin(n, galois::MethodFlag::UNPROTECTED),
            //        ee = graph.in_edge_end(n, galois::MethodFlag::UNPROTECTED); ii != ee; ++ii) {
            //   GNode dst = graph.getInEdgeDst(ii);
            //   SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
            //   if (ddata.dist + 1 == sdata.dist)
            //     sdata.numPaths += ddata.numPaths;
            // }
          },
          galois::loopname("BFS"), galois::steal());
      //galois::do_all(*levels.back(), Counter(graph), "COUNTER", true);
    }
    delete levels.back();
    levels.pop_back();

    Tbfs.stop();
  }

  void CountPaths(Graph& graph) {

    galois::do_all(galois::iterate(graph),
        [&] (GNode n) {
          auto& sdata = graph.getData(n, galois::MethodFlag::UNPROTECTED);
          unsigned long np = 0;
          for (Graph::in_edge_iterator ii = graph.in_edge_begin(n, galois::MethodFlag::UNPROTECTED),
                 ee = graph.in_edge_end(n, galois::MethodFlag::UNPROTECTED); ii != ee; ++ii) {
            GNode dst = graph.getInEdgeDst(ii);
            SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
            if (ddata.dist + 1 == sdata.dist)
              np += ddata.numPaths;
          }
          sdata.numPaths = sdata.numPaths + np;

        },
        galois::loopname("CountPaths"));
  }

  template <typename Deq>
  void ComputeDep(Graph& graph, Deq& levels) {

    galois::StatTimer Tdep("DepTime");
    Tdep.start();

    for (int i = levels.size() - 1; i > 0; --i)
      galois::do_all(galois::iterate(*levels[i-1]), 
          [&] (GNode n) {
            SNode& sdata = graph.getData(n, galois::MethodFlag::UNPROTECTED);
            for (auto ii : graph.edges(n, galois::MethodFlag::UNPROTECTED)) {
              GNode dst = graph.getEdgeDst(ii);
              SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
              if (ddata.dist == sdata.dist + 1)
                sdata.dependencies += ((float)sdata.numPaths / (float)ddata.numPaths) * (1 + ddata.dependencies);
            }
          },
          galois::loopname("ComputeDep"), galois::steal());
    Tdep.stop();
  }

  void operator()(Graph& graph, GNode source) {
    galois::StatTimer Tlevel("LevelTime");
    galois::StatTimer Tcount("CountTime");

    Initialize(graph);


    std::deque<Bag*> levels;
    BFS(graph, source, levels);

    ComputeDep(graph, levels);


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
    const unsigned MAX_COUNT = 20;

    std::cout << "Verification not implemented" << std::endl;
    std::cout << "Printing first " << MAX_COUNT << " values instead" << std::endl;

    unsigned count = 0;
    for (typename Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei && count < MAX_COUNT; ++ii, ++count) {
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
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer T("TotalTime");

  T.start();
  switch (algo) {
    case Algo::async: run<AsyncAlgo>();     break;
    case Algo::leveled: run<LeveledAlgo>(); break;
  }
  T.stop();

  return 0;
}
