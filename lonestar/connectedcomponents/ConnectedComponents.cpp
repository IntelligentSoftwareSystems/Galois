/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
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
#include "galois/AtomicHelpers.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/UnionFind.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/OCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "galois/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/runtime/Profile.h"

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>

#include <ostream>
#include <fstream>

const char* name = "Connected Components";
const char* desc = "Computes the connected components of a graph";
const char* url  = 0;

enum Algo {
  async,
  edgeasync,
  edgetiledasync,
  blockedasync,
  labelProp,
  serial,
  synchronous
};

enum OutputEdgeType { void_, int32_, int64_ };

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFilename(cll::Positional, cll::desc("<input file (symmetric)>"),
                  cll::Required);
static cll::opt<std::string>
    largestComponentFilename("outputLargestComponent",
                             cll::desc("[output graph file]"), cll::init(""));
static cll::opt<std::string>
    permutationFilename("outputNodePermutation",
                        cll::desc("[output node permutation file]"),
                        cll::init(""));
cll::opt<unsigned int>
    memoryLimit("memoryLimit",
                cll::desc("Memory limit for out-of-core algorithms (in MB)"),
                cll::init(~0U));
static cll::opt<OutputEdgeType> writeEdgeType(
    "edgeType", cll::desc("Input/Output edge type:"),
    cll::values(
        clEnumValN(OutputEdgeType::void_, "void", "no edge values"),
        clEnumValN(OutputEdgeType::int32_, "int32", "32 bit edge values"),
        clEnumValN(OutputEdgeType::int64_, "int64", "64 bit edge values"),
        clEnumValEnd),
    cll::init(OutputEdgeType::void_));
static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm:"),
    cll::values(clEnumValN(Algo::async, "Async", "Asynchronous"),
                clEnumValN(Algo::edgeasync, "EdgeAsync", "Edge-Asynchronous"),
                clEnumValN(Algo::edgetiledasync, "EdgetiledAsync",
                           "EdgeTiled-Asynchronous (default)"),
                clEnumValN(Algo::blockedasync, "BlockedAsync",
                           "Blocked asynchronous"),
                clEnumValN(Algo::labelProp, "LabelProp",
                           "Using label propagation algorithm"),
                clEnumValN(Algo::serial, "Serial", "Serial"),
                clEnumValN(Algo::synchronous, "Sync", "Synchronous"),

                clEnumValEnd),
    cll::init(Algo::edgetiledasync));

struct Node : public galois::UnionFindNode<Node> {
  using component_type = Node*;

  Node() : galois::UnionFindNode<Node>(const_cast<Node*>(this)) {}
  Node(const Node& o) : galois::UnionFindNode<Node>(o.m_component) {}

  Node& operator=(const Node& o) {
    Node c(o);
    std::swap(c, *this);
    return *this;
  }

  component_type component() { return this->findAndCompress(); }
  bool isRepComp(unsigned int x) { return false; }
};

const unsigned int LABEL_INF = std::numeric_limits<unsigned int>::max();

/**
 * Serial connected components algorithm. Just use union-find.
 */
struct SerialAlgo {
  using Graph =
      galois::graphs::LC_CSR_Graph<Node, void>::with_no_lockable<true>::type;
  using GNode = Graph::GraphNode;

  template <typename G>
  void readGraph(G& graph) {
    galois::graphs::readGraph(graph, inputFilename);
  }

  void operator()(Graph& graph) {

    for (const GNode& src : graph) {

      Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);

      for (auto ii : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
        GNode dst   = graph.getEdgeDst(ii);
        Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        sdata.merge(&ddata);
      }
    }
  }
};

/**
 * Synchronous connected components algorithm.  Initially all nodes are in
 * their own component. Then, we merge endpoints of edges to form the spanning
 * tree. Merging is done in two phases to simplify concurrent updates: (1)
 * find components and (2) union components.  Since the merge phase does not
 * do any finds, we only process a fraction of edges at a time; otherwise,
 * the union phase may unnecessarily merge two endpoints in the same
 * component.
 */
struct SynchronousAlgo {
  using Graph =
      galois::graphs::LC_CSR_Graph<Node, void>::with_no_lockable<true>::type;
  using GNode = Graph::GraphNode;

  template <typename G>
  void readGraph(G& graph) {
    galois::graphs::readGraph(graph, inputFilename);
  }

  struct Edge {
    GNode src;
    Node* ddata;
    int count;
    Edge(GNode src, Node* ddata, int count)
        : src(src), ddata(ddata), count(count) {}
  };

  void operator()(Graph& graph) {
    size_t rounds = 0;
    galois::GAccumulator<size_t> emptyMerges;

    galois::InsertBag<Edge> wls[2];
    galois::InsertBag<Edge>* next;
    galois::InsertBag<Edge>* cur;

    cur  = &wls[0];
    next = &wls[1];

    galois::do_all(galois::iterate(graph), [&](const GNode& src) {
      for (auto ii : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
        GNode dst = graph.getEdgeDst(ii);
        if (src >= dst)
          continue;
        Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        cur->push(Edge(src, &ddata, 0));
        break;
      }
    });

    while (!cur->empty()) {
      galois::do_all(galois::iterate(*cur),
                     [&](const Edge& edge) {
                       Node& sdata = graph.getData(
                           edge.src, galois::MethodFlag::UNPROTECTED);
                       if (!sdata.merge(edge.ddata))
                         emptyMerges += 1;
                     },
                     galois::loopname("Merge"));

      galois::do_all(
          galois::iterate(*cur),
          [&](const Edge& edge) {
            GNode src   = edge.src;
            Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
            Node* scomponent = sdata.findAndCompress();
            Graph::edge_iterator ii =
                graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
            Graph::edge_iterator ei =
                graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
            int count = edge.count + 1;
            std::advance(ii, count);
            for (; ii != ei; ++ii, ++count) {
              GNode dst = graph.getEdgeDst(ii);
              if (src >= dst)
                continue;
              Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
              Node* dcomponent = ddata.findAndCompress();
              if (scomponent != dcomponent) {
                next->push(Edge(src, dcomponent, count));
                break;
              }
            }
          },
          galois::loopname("Find"));

      cur->clear();
      std::swap(cur, next);
      rounds += 1;
    }

    galois::runtime::reportStat_Single("CC-Sync", "rounds", rounds);
    galois::runtime::reportStat_Single("CC-Sync", "emptyMerges",
                                       emptyMerges.reduce());
  }
};

struct LabelPropAlgo {

  struct LNode {
    using component_type = unsigned int;
    std::atomic<unsigned int> comp_current;
    unsigned int comp_old;

    component_type component() { return comp_current; }
    bool isRep() { return false; }
    bool isRepComp(unsigned int x) { return x == comp_current; }
  };

  using Graph =
      galois::graphs::LC_CSR_Graph<LNode, void>::with_no_lockable<true>::type;

  using GNode          = Graph::GraphNode;
  using component_type = LNode::component_type;

  template <typename G>
  void readGraph(G& graph) {
    galois::graphs::readGraph(graph, inputFilename);
  }

  void operator()(Graph& graph) {
    galois::GReduceLogicalOR changed;
    do {
      changed.reset();
      galois::do_all(
          galois::iterate(graph),
          [&](const GNode& src) {
            LNode& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
            if (sdata.comp_old > sdata.comp_current) {
              sdata.comp_old = sdata.comp_current;

              changed.update(true);

              for (auto e : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
                GNode dst              = graph.getEdgeDst(e);
                auto& ddata            = graph.getData(dst);
                unsigned int label_new = sdata.comp_current;
                galois::atomicMin(ddata.comp_current, label_new);
              }
            }
          },
          galois::no_conflicts(), galois::steal(),
          galois::loopname("LabelPropAlgo"));

    } while (changed.reduce());
  }
};

/**
 * Like synchronous algorithm, but if we restrict path compression (as done is
 * @link{UnionFindNode}), we can perform unions and finds concurrently.
 */
struct AsyncAlgo {
  using Graph =
      galois::graphs::LC_CSR_Graph<Node, void>::with_no_lockable<true>::type;
  using GNode = Graph::GraphNode;

  template <typename G>
  void readGraph(G& graph) {
    galois::graphs::readGraph(graph, inputFilename);
  }

  void operator()(Graph& graph) {
    galois::GAccumulator<size_t> emptyMerges;

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);

          for (auto ii : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
            GNode dst   = graph.getEdgeDst(ii);
            Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

            if (src >= dst)
              continue;

            if (!sdata.merge(&ddata))
              emptyMerges += 1;
          }
        },
        galois::loopname("CC-Async"));

    galois::runtime::reportStat_Single("CC-Async", "emptyMerges",
                                       emptyMerges.reduce());
  }
};

struct EdgeTiledAsyncAlgo {
  using Graph =
      galois::graphs::LC_CSR_Graph<Node, void>::with_no_lockable<true>::type;
  using GNode = Graph::GraphNode;

  template <typename G>
  void readGraph(G& graph) {
    galois::graphs::readGraph(graph, inputFilename);
  }

  struct EdgeTile {
    // Node* sData;
    GNode src;
    Graph::edge_iterator beg;
    Graph::edge_iterator end;
  };

  /*struct EdgeTileMaker {
      EdgeTile operator() (Node* sdata, Graph::edge_iterator beg,
  Graph::edge_iterator end) const{ return EdgeTile{sdata, beg, end};
      }
  };*/

  const int EDGE_TILE_SIZE = 512; // 512 -> 64
  void operator()(Graph& graph) {
    galois::GAccumulator<size_t> emptyMerges;

    galois::InsertBag<EdgeTile> works;

    const int CHUNK_SIZE = 1;
    std::cout << "INFO: Using edge tile size of " << EDGE_TILE_SIZE
              << " and chunk size of " << CHUNK_SIZE << "\n";
    std::cout << "WARNING: Performance varies considerably due to parameter.\n";
    std::cout
        << "WARNING: Do not expect the default to be good for your graph.\n";

    galois::do_all(galois::iterate(graph),
                   [&](const GNode& src) {
                     // Node& sdata=graph.getData(src,
                     // galois::MethodFlag::UNPROTECTED);
                     auto beg =
                         graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
                     const auto end =
                         graph.edge_end(src, galois::MethodFlag::UNPROTECTED);

                     assert(beg <= end);
                     if ((end - beg) > EDGE_TILE_SIZE) {
                       for (; beg + EDGE_TILE_SIZE < end;) {
                         auto ne = beg + EDGE_TILE_SIZE;
                         assert(ne < end);
                         works.push_back(EdgeTile{src, beg, ne});
                         beg = ne;
                       }
                     }

                     if ((end - beg) > 0) {
                       works.push_back(EdgeTile{src, beg, end});
                     }
                   },
                   galois::loopname("CC-EdgeTiledAsyncInit"), galois::steal());

    galois::do_all(
        galois::iterate(works),
        [&](const EdgeTile& tile) {
          // Node& sdata = *(tile.sData);
          GNode src   = tile.src;
          Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);

          for (auto ii = tile.beg; ii != tile.end; ++ii) {
            GNode dst = graph.getEdgeDst(ii);
            if (src >= dst)
              continue;

            Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

            if (src >= dst)
              continue;

            if (!sdata.merge(&ddata))
              emptyMerges += 1;
          }
        },
        galois::loopname("CC-edgetiledAsync"), galois::steal(),
        galois::chunk_size<CHUNK_SIZE>() // 16 -> 1
    );

    galois::runtime::reportStat_Single("CC-edgeTiledAsync", "emptyMerges",
                                       emptyMerges.reduce());
  }
};

struct EdgeAsyncAlgo {
  using Graph =
      galois::graphs::LC_CSR_Graph<Node, void>::with_no_lockable<true>::type;
  using GNode = Graph::GraphNode;
  using Edge  = std::pair<GNode, typename Graph::edge_iterator>;

  template <typename G>
  void readGraph(G& graph) {
    galois::graphs::readGraph(graph, inputFilename);
  }

  void operator()(Graph& graph) {
    galois::GAccumulator<size_t> emptyMerges;

    galois::InsertBag<Edge> works;

    galois::do_all(galois::iterate(graph),
                   [&](const GNode& src) {
                     for (auto ii :
                          graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
                       if (src < graph.getEdgeDst(ii)) {
                         works.push_back(std::make_pair(src, ii));
                       }
                     }
                   },
                   galois::loopname("CC-EdgeAsyncInit"), galois::steal());

    galois::do_all(
        galois::iterate(works),
        [&](Edge& e) {
          Node& sdata = graph.getData(e.first, galois::MethodFlag::UNPROTECTED);
          GNode dst   = graph.getEdgeDst(e.second);
          Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

          if (e.first > dst)
            // continue;
            ;
          else if (!sdata.merge(&ddata)) {
            emptyMerges += 1;
          }
        },
        galois::loopname("CC-EdgeAsync"), galois::steal());

    galois::runtime::reportStat_Single("CC-Async", "emptyMerges",
                                       emptyMerges.reduce());
  }
};

/**
 * Improve performance of async algorithm by following machine topology.
 */
struct BlockedAsyncAlgo {
  using Graph =
      galois::graphs::LC_CSR_Graph<Node, void>::with_no_lockable<true>::type;
  using GNode = Graph::GraphNode;

  struct WorkItem {
    GNode src;
    Graph::edge_iterator start;
  };

  template <typename G>
  void readGraph(G& graph) {
    galois::graphs::readGraph(graph, inputFilename);
  }

  //! Add the next edge between components to the worklist
  template <bool MakeContinuation, int Limit, typename Pusher>
  static void process(Graph& graph, const GNode& src,
                      const Graph::edge_iterator& start, Pusher& pusher) {

    Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
    int count   = 1;
    for (Graph::edge_iterator
             ii = start,
             ei = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
         ii != ei; ++ii, ++count) {
      GNode dst   = graph.getEdgeDst(ii);
      Node& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

      if (src >= dst)
        continue;

      if (sdata.merge(&ddata)) {
        if (Limit == 0 || count != Limit)
          continue;
      }

      if (MakeContinuation || (Limit != 0 && count == Limit)) {
        WorkItem item = {src, ii + 1};
        pusher.push(item);
        break;
      }
    }
  }

  void operator()(Graph& graph) {
    galois::InsertBag<WorkItem> items;

    galois::do_all(galois::iterate(graph),
                   [&](const GNode& src) {
                     Graph::edge_iterator start =
                         graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
                     if (galois::substrate::ThreadPool::getSocket() == 0) {
                       process<true, 0>(graph, src, start, items);
                     } else {
                       process<true, 1>(graph, src, start, items);
                     }
                   },
                   galois::loopname("Initialize"));

    galois::for_each(galois::iterate(items),
                     [&](const WorkItem& item, auto& ctx) {
                       process<true, 0>(graph, item.src, item.start, ctx);
                     },
                     galois::loopname("Merge"),
                     galois::wl<galois::worklists::PerSocketChunkFIFO<128>>());
  }
};

template <typename Graph>
bool verify(
    Graph& graph,
    typename std::enable_if<galois::graphs::is_segmented<Graph>::value>::type* =
        0) {
  return true;
}

template <typename Graph>
bool verify(Graph& graph,
            typename std::enable_if<
                !galois::graphs::is_segmented<Graph>::value>::type* = 0) {

  using GNode = typename Graph::GraphNode;

  auto is_bad = [&graph](const GNode& n) {
    auto& me = graph.getData(n);
    for (auto ii : graph.edges(n)) {
      GNode dst  = graph.getEdgeDst(ii);
      auto& data = graph.getData(dst);
      if (data.component() != me.component()) {
        std::cerr << "not in same component: " << (unsigned int)n << " ("
                  << me.component() << ")"
                  << " and " << (unsigned int)dst << " (" << data.component()
                  << ")"
                  << "\n";
        return true;
      }
    }
    return false;
  };

  return galois::ParallelSTL::find_if(graph.begin(), graph.end(), is_bad) ==
         graph.end();
}

template <typename Algo, typename Graph>
typename Graph::node_data_type::component_type findLargest(Graph& graph) {

  using GNode          = typename Graph::GraphNode;
  using component_type = typename Graph::node_data_type::component_type;

  using ReducerMap =
      galois::GMapPerItemReduce<component_type, int, std::plus<int>>;
  using Map = typename ReducerMap::container_type;

  using ComponentSizePair = std::pair<component_type, int>;

  ReducerMap accumMap;
  galois::GAccumulator<size_t> accumReps;

  galois::do_all(galois::iterate(graph),
                 [&](const GNode& x) {
                   auto& n = graph.getData(x, galois::MethodFlag::UNPROTECTED);

                   if (std::is_same<Algo, LabelPropAlgo>::value) {
                     if (n.isRepComp((unsigned int)x)) {
                       accumReps += 1;
                       return;
                     }
                   } else {
                     if (n.isRep()) {
                       accumReps += 1;
                       return;
                     }
                   }

                   // Don't add reps to table to avoid adding components of size
                   // 1
                   accumMap.update(n.component(), 1);
                 },
                 galois::loopname("CountLargest"));

  Map& map    = accumMap.reduce();
  size_t reps = accumReps.reduce();

  auto sizeMax = [](const ComponentSizePair& a, const ComponentSizePair& b) {
    if (a.second > b.second) {
      return a;
    }
    return b;
  };

  using MaxComp =
      galois::GSimpleReducible<decltype(sizeMax), ComponentSizePair>;
  MaxComp maxComp(sizeMax);

  galois::do_all(galois::iterate(map),
                 [&](const ComponentSizePair& x) { maxComp.update(x); });

  ComponentSizePair largest = maxComp.reduce();

  // Compensate for dropping representative node of components
  double ratio       = graph.size() - reps + map.size();
  size_t largestSize = largest.second + 1;
  if (ratio)
    ratio = largestSize / ratio;

  std::cout << "Total components: " << reps << "\n";
  std::cout << "Number of non-trivial components: " << map.size()
            << " (largest size: " << largestSize << " [" << ratio << "])\n";

  return largest.first;
}

template <typename Graph>
void initialize(Graph& graph) {}

template <>
void initialize<LabelPropAlgo::Graph>(typename LabelPropAlgo::Graph& graph) {
  unsigned int id = 0;

  for (typename LabelPropAlgo::Graph::iterator ii = graph.begin(),
                                               ei = graph.end();
       ii != ei; ++ii, ++id) {
    graph.getData(*ii).comp_current = id;
    graph.getData(*ii).comp_old     = LABEL_INF;
  }
}

template <typename Algo>
void run() {
  using Graph = typename Algo::Graph;

  Algo algo;
  Graph graph;

  algo.readGraph(graph);
  std::cout << "Read " << graph.size() << " nodes\n";

  initialize(graph);

  galois::preAlloc(numThreads +
                   (3 * graph.size() * sizeof(typename Graph::node_data_type)) /
                       galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  galois::StatTimer T;
  T.start();
  algo(graph);
  T.stop();

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify || largestComponentFilename != "" ||
      permutationFilename != "") {
    findLargest<Algo, Graph>(graph);
    if (!verify(graph)) {
      GALOIS_DIE("verification failed");
    }
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer T("TotalTime");
  T.start();
  switch (algo) {
  case Algo::async:
    run<AsyncAlgo>();
    break;
  case Algo::edgeasync:
    run<EdgeAsyncAlgo>();
    break;
  case Algo::edgetiledasync:
    run<EdgeTiledAsyncAlgo>();
    break;
  case Algo::blockedasync:
    run<BlockedAsyncAlgo>();
    break;
  case Algo::labelProp:
    run<LabelPropAlgo>();
    break;
  case Algo::serial:
    run<SerialAlgo>();
    break;
  case Algo::synchronous:
    run<SynchronousAlgo>();
    break;

  default:
    std::cerr << "Unknown algorithm\n";
    abort();
  }
  T.stop();

  return 0;
}
