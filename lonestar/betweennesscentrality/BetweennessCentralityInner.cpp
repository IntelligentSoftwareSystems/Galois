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
 * Betweenness centrality. Implementation based on Ligra. Does BC with only
 * a single source. "Inner" because parallelism occurs within a source
 * calculation rather than with sources themselves.
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
static const char* desc = "Single source betweeness-centrality with a parallel "
                          "inner loop";
static const char* url = 0;

// algorithm types supported
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

/**
 * Load an in-out (graph with both in and out edges) graph into memory.
 *
 * Since it has both in and out edges, a symmetric graph or a graph AND its
 * transpose must be provided by the user.
 *
 * @tparam Graph Should be an InOut graph type
 * @param graph Graph object to load the graph into
 */
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

/**
 * Initialize the graph by loading it and get the source to initiate BC
 * from.
 *
 * @tparam Algo algorithm to load graph for
 * @param graph object to load graph into
 * @returns Source to start bc from
 */
template<typename Algo>
typename Algo::Graph::GraphNode initialize(typename Algo::Graph& graph) {
  readInOutGraph(graph);
  galois::gPrint("Read ", graph.size(), " nodes\n");

  if (startNode >= graph.size()) {
    galois::gError("Failed to get source: ", startNode);
    assert(0);
    abort();
  }
  
  // get source 
  typename Algo::Graph::iterator it = graph.begin();
  std::advance(it, startNode);
  return *it;
}

// vars for use in worklist in AsyncAlgo
static const int ChunkSize = 128;
static const int bfsChunkSize = 64;

/**
 * Contains code for running the asynchronous version of BC-Inner
 */
struct AsyncAlgo {
  /**
   * Data held at each node in the graph.
   */
  struct SNode {
    int numPaths;
    float dependencies;
    int dist;

    SNode() 
      : numPaths(-std::numeric_limits<int>::max()), 
        dependencies(-std::numeric_limits<float>::max()), 
        dist(std::numeric_limits<int>::max()) { }
  };

  using InnerGraph = galois::graphs::LC_CSR_Graph<SNode, void>
                     ::with_no_lockable<true>::type 
                     ::with_numa_alloc<true>::type;
  using Graph = galois::graphs::LC_InOut_Graph<InnerGraph>;
  using GNode = Graph::GraphNode; 

  std::string name() const { return "async"; }

  /**
   * Initialize all node data in the graph to default values.
   *
   * @param graph Graph to initialize
   */
  void Initialize(Graph& graph) {
    galois::do_all(
      galois::iterate(graph),
      [&] (GNode n) {
        SNode& data = graph.getData(n, galois::MethodFlag::UNPROTECTED);
        data.numPaths = -std::numeric_limits<int>::max();
        data.dependencies = -std::numeric_limits<float>::max();
        data.dist = std::numeric_limits<int>::max();
      },
      galois::loopname("Initialize"));
  }

  /**
   * Get the number of shortests paths on each node by looping through
   * predecessors on the shortest path DAG and adding their short path
   * count to self.
   *
   * @param graph graph object to operate on
   */
  void CountPaths(Graph& graph) {
    // Lambda function to get distance stored on a node
    auto indexer = [&] (const GNode& n) {
      return graph.getData(n, galois::MethodFlag::UNPROTECTED).dist;
    };

    using OBIM = galois::worklists::OrderedByIntegerMetric<
                   decltype(indexer), 
                   galois::worklists::dChunkedFIFO<ChunkSize>
                 >;

    galois::for_each(
      galois::iterate(graph),
      [&] (GNode n, auto& ctx) {
        SNode& sdata = graph.getData(n, galois::MethodFlag::UNPROTECTED);

        while (sdata.numPaths == -std::numeric_limits<int>::max()) {
          int curNumPaths = 0;
          bool allReady = true;

          // loop through this node's predecessors in the BFS DAG and add their
          // number of shortest paths to this node's current count ONLY
          // if it is finalized there; else wait until they are
          for (auto ii = graph.in_edge_begin(n, galois::MethodFlag::UNPROTECTED),
                    ee = graph.in_edge_end(n, galois::MethodFlag::UNPROTECTED); 
               ii != ee;
               ++ii) {
            GNode dst = graph.getInEdgeDst(ii);
            SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

            if (ddata.dist + 1 == sdata.dist) {
              if (ddata.numPaths != -std::numeric_limits<int>::max()) {
                curNumPaths += ddata.numPaths;
              } else {
                allReady = false;
              }
            }
          }

          // finalize only if all predecessors had a finalized numPaths
          if (allReady) sdata.numPaths = curNumPaths;
        } // end while
      },
      galois::no_conflicts(),
      galois::no_pushes(),
      galois::wl<OBIM>(indexer),
      galois::loopname("CountPaths")
    );
  }

  /**
   * Loop through successors to use finalized dependency values to calculate
   * own dependency.
   *
   * @param graph graph object to operate on
   */
  void ComputeDep(Graph& graph) {
    // Lambda function to essentially get the negative distance on a node
    // (i.e. smaller distances = bigger index)
    // Used by OBIM to prioritize based on highest distance
    auto indexer = [&] (const GNode& n) {
      return std::numeric_limits<int>::max() - 
             graph.getData(n, galois::MethodFlag::UNPROTECTED).dist;
    };

    using OBIM = galois::worklists::OrderedByIntegerMetric<
                   decltype(indexer),
                   galois::worklists::dChunkedFIFO<ChunkSize>
                 >;

    galois::for_each(
      galois::iterate(graph),
      [&] (GNode n, auto& ctx) {
        SNode& sdata = graph.getData(n, galois::MethodFlag::UNPROTECTED);

        while (sdata.dependencies == -std::numeric_limits<float>::max()) {
          float newDep = 0.0;
          bool allReady = true;

          // loop through successors and grab dependency value for use if 
          // finalized; if not all dependencies finalized, then reloop until 
          // they are
          for (auto ii : graph.edges(n, galois::MethodFlag::UNPROTECTED)) {
            GNode dst = graph.getEdgeDst(ii);
            SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

            if (ddata.dist == sdata.dist + 1) {
              if (ddata.dependencies != -std::numeric_limits<float>::max()) {
                newDep += ((float)sdata.numPaths / (float)ddata.numPaths) * 
                          (1 + ddata.dependencies);
              } else {
                allReady = false;
              }
            }
          }

          // only do this if all successors have finalized dep value
          if (allReady) sdata.dependencies = newDep;
        } 
      },
      galois::no_conflicts(),
      galois::no_pushes(),
      galois::wl<OBIM>(indexer),
      galois::loopname("ComputeDep")
    );
  }

  /**
   * Run all steps of single-source BC
   */
  void operator()(Graph& graph, GNode source) {
    Initialize(graph);

    galois::StatTimer Tbfs("Tbfs", "BCAsync");

    Tbfs.start();
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
    SNode() : numPaths(~0UL), dependencies(-std::numeric_limits<float>::max()), 
              dist(std::numeric_limits<int>::max()) { }
  };

  using InnerGraph = galois::graphs::LC_CSR_Graph<SNode,void>
                       ::with_no_lockable<true>::type 
                       ::with_numa_alloc<true>::type;
  using Graph = galois::graphs::LC_InOut_Graph<InnerGraph>;
  using GNode = Graph::GraphNode;
  using Bag = galois::InsertBag<GNode>;

  std::string name() const { return "Leveled"; }

  /**
   * Initialize node fields to default value
   *
   * @param graph Graph to operate on
   */
  void Initialize(Graph& graph) {
    galois::do_all(
      galois::iterate(graph),
      [&] (GNode n) {
        SNode& data = graph.getData(n, galois::MethodFlag::UNPROTECTED);
        data.numPaths = 0;
        data.dependencies = 0.0; //std::numeric_limits<float>::lowest();
        data.dist = std::numeric_limits<int>::max();
      },
      galois::loopname("Initialize"));
  }

  /**
   * Do BFS level-by-level, push style. Also calculate the number of shortest
   * paths as you go along.
   *
   * @tparam Deq type of the container that holds levels
   *
   * @param graph Graph to operate on
   * @param source Source node for BFS
   * @param levels Container which holds the levels on the BFS DAG each node
   * is on
   */
  template <typename Deq>
  void BFS(Graph& graph, GNode source, Deq& levels) {
    galois::StatTimer Tbfs("BFSTime", "BCLeveled");
    Tbfs.start();

    levels.push_back(new Bag());
    levels[0]->push_back(source);
    graph.getData(source).dist = 0;
    graph.getData(source).numPaths = 1;

    while (!levels.back()->empty()) {
      Bag* b = levels.back();
      levels.push_back(new Bag());
      Bag* newBag = levels.back();

      galois::do_all(
        galois::iterate(*b), 
        [&] (GNode n) {
          auto& sdata = graph.getData(n, galois::MethodFlag::UNPROTECTED);

          for (auto ii : graph.edges(n, galois::MethodFlag::UNPROTECTED)) {
            GNode dst = graph.getEdgeDst(ii);
            SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

            if (ddata.dist.load(std::memory_order_relaxed) == 
                  std::numeric_limits<int>::max()) {
              if (std::numeric_limits<int>::max() == 
                    ddata.dist.exchange(sdata.dist + 1)) {
                newBag->push_back(dst);
              }

              ddata.numPaths += sdata.numPaths;
            } else if (ddata.dist == sdata.dist + 1) {
              ddata.numPaths += sdata.numPaths;
            }
          }
        },
        galois::loopname("BFS"), 
        galois::steal()
      );
    }

    delete levels.back(); // get rid of the last level done since it's empty
    levels.pop_back();

    Tbfs.stop();
  }

  /**
   * Compute dependencies level by level going backwards as dependencies
   * are calculated from successors.
   *
   * @tparam Deq type of the container that holds levels
   *
   * @param graph Graph to operate on
   * @param levels Container which holds the levels on the BFS DAG each node
   * is on
   */
  template <typename Deq>
  void ComputeDep(Graph& graph, Deq& levels) {
    galois::StatTimer Tdep("DepTime");
    Tdep.start();
    galois::gInfo("Size is ", levels.size());

    // start from next to last level and go to level before level 0:
    // last level is all leaf nodes, so dependency should remain at 0,
    // and level 0 only has the source node, which should not be updating
    // its dependency
    for (int i = levels.size() - 2; i > 0; --i) {
      galois::gInfo(i);
      galois::do_all(
        galois::iterate(*levels[i]), 
        [&] (GNode n) {
          SNode& sdata = graph.getData(n, galois::MethodFlag::UNPROTECTED);

          // grab dependencies from successors in DAG
          for (auto ii : graph.edges(n, galois::MethodFlag::UNPROTECTED)) {
            GNode dst = graph.getEdgeDst(ii);
            SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

            if (ddata.dist == sdata.dist + 1) {
              sdata.dependencies += 
                ((float)sdata.numPaths / (float)ddata.numPaths) * 
                  (1 + ddata.dependencies);
            }
          }
        },
        galois::loopname("ComputeDep"),
        galois::steal()
      );
    }
    Tdep.stop();
  }

  /**
   * Do BFS (and get shortest paths), then get dependency value from it.
   */
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

/**
 * Common run code to the 2 algorithms.
 *
 * @tparam Algo the algorithm type to run
 */
template<typename Algo>
void run() {
  using Graph = typename Algo::Graph;
  using GNode = typename Graph::GraphNode;

  Algo algo;
  Graph graph;
  GNode source = initialize<Algo>(graph);

  galois::reportPageAlloc("MeminfoPre");
  galois::preAlloc(numThreads + 
                     (3 * graph.size() * sizeof(typename Graph::node_data_type)) /
                       galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoMid");

  galois::StatTimer T;
  galois::gPrint("Running ", algo.name(), " version\n");

  T.start();
  algo(graph, source);
  T.stop();
  
  galois::reportPageAlloc("MeminfoPost");

  // prints stats for first MAX_COUNT nodes
  if (!skipVerify) {
    const unsigned MAX_COUNT = 20;

    galois::gPrint("Verification not implemented\n");
    galois::gPrint("Printing first ", MAX_COUNT, " values instead\n");

    unsigned count = 0;
    for (typename Graph::iterator ii = graph.begin(), 
                                  ei = graph.end(); 
         ii != ei && count < MAX_COUNT; 
         ++ii, ++count) {
      std::cout << count << " "
                << std::setiosflags(std::ios::fixed) << std::setprecision(6) 
                << graph.getData(*ii).dependencies
                << " " << graph.getData(*ii).numPaths 
                << " " << graph.getData(*ii).dist
                << "\n";
    }
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
