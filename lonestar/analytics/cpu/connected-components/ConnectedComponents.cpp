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
#include "galois/AtomicHelpers.h"
#include "galois/Bag.h"
#include "galois/ParallelSTL.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/UnionFind.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/OCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "galois/runtime/Profile.h"
#include "Lonestar/BoilerPlate.h"

#include "llvm/Support/CommandLine.h"

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>

#include <ostream>
#include <fstream>

const char* name = "Connected Components";
const char* desc = "Computes the connected components of a graph";

namespace cll = llvm::cl;

enum Algo {
  serial,
  labelProp,
  synchronous,
  async,
  edgeasync,
  blockedasync,
  edgetiledasync,
  afforest,
  edgeafforest,
  edgetiledafforest,
};

static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm:"),
    cll::values(
        clEnumValN(Algo::async, "Async", "Asynchronous"),
        clEnumValN(Algo::edgeasync, "EdgeAsync", "Edge-Asynchronous"),
        clEnumValN(Algo::edgetiledasync, "EdgetiledAsync",
                   "EdgeTiled-Asynchronous (default)"),
        clEnumValN(Algo::blockedasync, "BlockedAsync", "Blocked asynchronous"),
        clEnumValN(Algo::labelProp, "LabelProp",
                   "Using label propagation algorithm"),
        clEnumValN(Algo::serial, "Serial", "Serial"),
        clEnumValN(Algo::synchronous, "Sync", "Synchronous"),
        clEnumValN(Algo::afforest, "Afforest", "Using Afforest sampling"),
        clEnumValN(Algo::edgeafforest, "EdgeAfforest",
                   "Using Afforest sampling, Edge-wise"),
        clEnumValN(Algo::edgetiledafforest, "EdgetiledAfforest",
                   "Using Afforest sampling, EdgeTiled")

            ),
    cll::init(Algo::edgetiledasync));

static cll::opt<std::string>
    largestComponentFilename("outputLargestComponent",
                             cll::desc("[output graph file]"), cll::init(""));
static cll::opt<std::string>
    permutationFilename("outputNodePermutation",
                        cll::desc("[output node permutation file]"),
                        cll::init(""));
#ifndef NDEBUG
enum OutputEdgeType { void_, int32_, int64_ };
static cll::opt<unsigned int>
    memoryLimit("memoryLimit",
                cll::desc("Memory limit for out-of-core algorithms (in MB)"),
                cll::init(~0U));
static cll::opt<OutputEdgeType> writeEdgeType(
    "edgeType", cll::desc("Input/Output edge type:"),
    cll::values(
        clEnumValN(OutputEdgeType::void_, "void", "no edge values"),
        clEnumValN(OutputEdgeType::int32_, "int32", "32 bit edge values"),
        clEnumValN(OutputEdgeType::int64_, "int64", "64 bit edge values")),
    cll::init(OutputEdgeType::void_));
#endif

// TODO (bozhi) LLVM commandline library now supports option categorization.
// Categorize params when libllvm is updated to make -help beautiful!
// static cll::OptionCategory ParamCat("Algorithm-Specific Parameters",
//                                       "Only used for specific algorithms.");
static cll::opt<uint32_t>
    EDGE_TILE_SIZE("edgeTileSize",
                   cll::desc("(For Edgetiled algos) Size of edge tiles "
                             "(default 512)"),
                   // cll::cat(ParamCat),
                   cll::init(512)); // 512 -> 64
static const int CHUNK_SIZE = 1;
//! parameter for the Vertex Neighbor Sampling step of Afforest algorithm
static cll::opt<uint32_t> NEIGHBOR_SAMPLES(
    "vns",
    cll::desc("(For Afforest and its variants) number of edges "
              "per vertice to process initially for exposing "
              "partial connectivity (default 2)"),
    // cll::cat(ParamCat),
    cll::init(2));
//! parameter for the Large Component Skipping step of Afforest algorithm
static cll::opt<uint32_t> COMPONENT_SAMPLES(
    "lcs",
    cll::desc("(For Afforest and its variants) number of times "
              "randomly sampling over vertices to approximately "
              "capture the largest intermediate component "
              "(default 1024)"),
    // cll::cat(ParamCat),
    cll::init(1024));

struct Node : public galois::UnionFindNode<Node> {
  using component_type = Node*;

  Node() : galois::UnionFindNode<Node>(const_cast<Node*>(this)) {}
  Node(const Node& o) : galois::UnionFindNode<Node>(o.m_component) {}

  Node& operator=(const Node& o) {
    Node c(o);
    std::swap(c, *this);
    return *this;
  }

  component_type component() { return this->get(); }
  bool isRepComp(unsigned int) { return false; }
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
    galois::graphs::readGraph(graph, inputFile);
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

    for (const GNode& src : graph) {
      Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      sdata.compress();
    }
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
    galois::graphs::readGraph(graph, inputFile);
  }

  void operator()(Graph& graph) {
    galois::GReduceLogicalOr changed;
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
          galois::disable_conflict_detection(), galois::steal(),
          galois::loopname("LabelPropAlgo"));
    } while (changed.reduce());
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
    galois::graphs::readGraph(graph, inputFile);
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
      galois::do_all(
          galois::iterate(*cur),
          [&](const Edge& edge) {
            Node& sdata =
                graph.getData(edge.src, galois::MethodFlag::UNPROTECTED);
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

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          sdata.compress();
        },
        galois::steal(), galois::loopname("Compress"));

    galois::runtime::reportStat_Single("CC-Sync", "rounds", rounds);
    galois::runtime::reportStat_Single("CC-Sync", "emptyMerges",
                                       emptyMerges.reduce());
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
    galois::graphs::readGraph(graph, inputFile);
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

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          sdata.compress();
        },
        galois::steal(), galois::loopname("CC-Async-Compress"));

    galois::runtime::reportStat_Single("CC-Async", "emptyMerges",
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
    galois::graphs::readGraph(graph, inputFile);
  }

  void operator()(Graph& graph) {
    galois::GAccumulator<size_t> emptyMerges;

    galois::InsertBag<Edge> works;

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          for (auto ii : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
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

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          sdata.compress();
        },
        galois::steal(), galois::loopname("CC-Async-Compress"));

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
    galois::graphs::readGraph(graph, inputFile);
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

    galois::do_all(
        galois::iterate(graph),
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

    galois::for_each(
        galois::iterate(items),
        [&](const WorkItem& item, auto& ctx) {
          process<true, 0>(graph, item.src, item.start, ctx);
        },
        galois::loopname("Merge"),
        galois::wl<galois::worklists::PerSocketChunkFIFO<128>>());

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          sdata.compress();
        },
        galois::steal(), galois::loopname("CC-Async-Compress"));
  }
};

struct EdgeTiledAsyncAlgo {
  using Graph =
      galois::graphs::LC_CSR_Graph<Node, void>::with_no_lockable<true>::type;
  using GNode = Graph::GraphNode;

  template <typename G>
  void readGraph(G& graph) {
    galois::graphs::readGraph(graph, inputFile);
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

  void operator()(Graph& graph) {
    galois::GAccumulator<size_t> emptyMerges;

    galois::InsertBag<EdgeTile> works;

    std::cout << "INFO: Using edge tile size of " << EDGE_TILE_SIZE
              << " and chunk size of " << CHUNK_SIZE << "\n";
    std::cout << "WARNING: Performance varies considerably due to parameter.\n";
    std::cout
        << "WARNING: Do not expect the default to be good for your graph.\n";

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          // Node& sdata=graph.getData(src,
          // galois::MethodFlag::UNPROTECTED);
          auto beg = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
          const auto end = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);

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
            if (!sdata.merge(&ddata))
              emptyMerges += 1;
          }
        },
        galois::loopname("CC-edgetiledAsync"), galois::steal(),
        galois::chunk_size<CHUNK_SIZE>() // 16 -> 1
    );

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          Node& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          sdata.compress();
        },
        galois::steal(), galois::loopname("CC-Async-Compress"));

    galois::runtime::reportStat_Single("CC-edgeTiledAsync", "emptyMerges",
                                       emptyMerges.reduce());
  }
};

template <typename component_type, typename Graph>
component_type approxLargestComponent(Graph& graph) {
  using map_type = std::unordered_map<
      component_type, int, std::hash<component_type>,
      std::equal_to<component_type>,
      galois::gstl::Pow2Alloc<std::pair<const component_type, int>>>;
  using pair_type = std::pair<component_type, int>;

  map_type comp_freq(COMPONENT_SAMPLES);
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<uint32_t> dist(0, graph.size() - 1);
  for (uint32_t i = 0; i < COMPONENT_SAMPLES; i++) {
    auto& ndata = graph.getData(dist(rng), galois::MethodFlag::UNPROTECTED);
    comp_freq[ndata.component()]++;
  }

  assert(!comp_freq.empty());
  auto most_frequent =
      std::max_element(comp_freq.begin(), comp_freq.end(),
                       [](const pair_type& a, const pair_type& b) {
                         return a.second < b.second;
                       });

  galois::gDebug(
      "Approximate largest intermediate component: ", most_frequent->first,
      " (hit rate ", 100.0 * (most_frequent->second) / COMPONENT_SAMPLES, "%)");

  return most_frequent->first;
}

/**
 * CC w/ Afforest sampling.
 *
 * [1] M. Sutton, T. Ben-Nun and A. Barak, "Optimizing Parallel Graph
 * Connectivity Computation via Subgraph Sampling," 2018 IEEE International
 * Parallel and Distributed Processing Symposium (IPDPS), Vancouver, BC, 2018,
 * pp. 12-21.
 */
struct AfforestAlgo {
  struct NodeData : public galois::UnionFindNode<NodeData> {
    using component_type = NodeData*;

    NodeData() : galois::UnionFindNode<NodeData>(const_cast<NodeData*>(this)) {}
    NodeData(const NodeData& o)
        : galois::UnionFindNode<NodeData>(o.m_component) {}

    component_type component() { return this->get(); }
    bool isRepComp(unsigned int) { return false; } // verify

  public:
    void link(NodeData* b) {
      NodeData* a = m_component.load(std::memory_order_relaxed);
      b           = b->m_component.load(std::memory_order_relaxed);
      while (a != b) {
        if (a < b)
          std::swap(a, b);
        // Now a > b
        NodeData* ac = a->m_component.load(std::memory_order_relaxed);
        if ((ac == a && a->m_component.compare_exchange_strong(a, b)) ||
            (b == ac))
          break;
        a = (a->m_component.load(std::memory_order_relaxed))
                ->m_component.load(std::memory_order_relaxed);
        b = b->m_component.load(std::memory_order_relaxed);
      }
    }
  };
  using Graph =
      galois::graphs::LC_CSR_Graph<NodeData,
                                   void>::with_no_lockable<true>::type;
  using GNode          = Graph::GraphNode;
  using component_type = NodeData::component_type;

  template <typename G>
  void readGraph(G& graph) {
    galois::graphs::readGraph(graph, inputFile);
  }

  void operator()(Graph& graph) {
    // (bozhi) should NOT go through single direction in sampling step: nodes
    // with edges less than NEIGHBOR_SAMPLES will fail
    for (uint32_t r = 0; r < NEIGHBOR_SAMPLES; ++r) {
      galois::do_all(
          galois::iterate(graph),
          [&](const GNode& src) {
            Graph::edge_iterator ii =
                graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
            Graph::edge_iterator ei =
                graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
            for (std::advance(ii, r); ii < ei; ii++) {
              GNode dst = graph.getEdgeDst(ii);
              NodeData& sdata =
                  graph.getData(src, galois::MethodFlag::UNPROTECTED);
              NodeData& ddata =
                  graph.getData(dst, galois::MethodFlag::UNPROTECTED);
              sdata.link(&ddata);
              break;
            }
          },
          galois::steal(), galois::loopname("Afforest-VNS-Link"));

      galois::do_all(
          galois::iterate(graph),
          [&](const GNode& src) {
            NodeData& sdata =
                graph.getData(src, galois::MethodFlag::UNPROTECTED);
            sdata.compress();
          },
          galois::steal(), galois::loopname("Afforest-VNS-Compress"));
    }

    galois::StatTimer StatTimer_Sampling("Afforest-LCS-Sampling");
    StatTimer_Sampling.start();
    const component_type c = approxLargestComponent<component_type>(graph);
    StatTimer_Sampling.stop();

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          NodeData& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          if (sdata.component() == c)
            return;
          Graph::edge_iterator ii =
              graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
          Graph::edge_iterator ei =
              graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
          for (std::advance(ii, NEIGHBOR_SAMPLES.getValue()); ii < ei; ++ii) {
            GNode dst = graph.getEdgeDst(ii);
            NodeData& ddata =
                graph.getData(dst, galois::MethodFlag::UNPROTECTED);
            sdata.link(&ddata);
          }
        },
        galois::steal(), galois::loopname("Afforest-LCS-Link"));

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          NodeData& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          sdata.compress();
        },
        galois::steal(), galois::loopname("Afforest-LCS-Compress"));
  }
};

/**
 * Edge CC w/ Afforest sampling
 */
struct EdgeAfforestAlgo {
  struct NodeData : public galois::UnionFindNode<NodeData> {
    using component_type = NodeData*;

    NodeData() : galois::UnionFindNode<NodeData>(const_cast<NodeData*>(this)) {}
    NodeData(const NodeData& o)
        : galois::UnionFindNode<NodeData>(o.m_component) {}

    component_type component() { return this->get(); }
    bool isRepComp(unsigned int) { return false; } // verify

  public:
    NodeData* hook_min(NodeData* b, NodeData* c = 0) {
      NodeData* a = m_component.load(std::memory_order_relaxed);
      b           = b->m_component.load(std::memory_order_relaxed);
      while (a != b) {
        if (a < b)
          std::swap(a, b);
        // Now a > b
        NodeData* ac = a->m_component.load(std::memory_order_relaxed);
        if (ac == a && a->m_component.compare_exchange_strong(a, b)) {
          if (b == c)
            return a; //! return victim
          return 0;
        }
        if (b == ac) {
          return 0;
        }
        a = (a->m_component.load(std::memory_order_relaxed))
                ->m_component.load(std::memory_order_relaxed);
        b = b->m_component.load(std::memory_order_relaxed);
      }
      return 0;
    }
  };
  using Graph =
      galois::graphs::LC_CSR_Graph<NodeData,
                                   void>::with_no_lockable<true>::type;
  using GNode          = Graph::GraphNode;
  using component_type = NodeData::component_type;

  using Edge = std::pair<GNode, GNode>;

  template <typename G>
  void readGraph(G& graph) {
    galois::graphs::readGraph(graph, inputFile);
  }

  void operator()(Graph& graph) {
    // (bozhi) should NOT go through single direction in sampling step: nodes
    // with edges less than NEIGHBOR_SAMPLES will fail
    for (uint32_t r = 0; r < NEIGHBOR_SAMPLES; ++r) {
      galois::do_all(
          galois::iterate(graph),
          [&](const GNode& src) {
            Graph::edge_iterator ii =
                graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
            Graph::edge_iterator ei =
                graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
            std::advance(ii, r);
            if (ii < ei) {
              GNode dst = graph.getEdgeDst(ii);
              NodeData& sdata =
                  graph.getData(src, galois::MethodFlag::UNPROTECTED);
              NodeData& ddata =
                  graph.getData(dst, galois::MethodFlag::UNPROTECTED);
              sdata.hook_min(&ddata);
            }
          },
          galois::steal(), galois::loopname("EdgeAfforest-VNS-Link"));
    }
    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          NodeData& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          sdata.compress();
        },
        galois::steal(), galois::loopname("EdgeAfforest-VNS-Compress"));

    galois::StatTimer StatTimer_Sampling("EdgeAfforest-LCS-Sampling");
    StatTimer_Sampling.start();
    const component_type c = approxLargestComponent<component_type>(graph);
    StatTimer_Sampling.stop();
    const component_type c0 =
        &(graph.getData(0, galois::MethodFlag::UNPROTECTED));

    galois::InsertBag<Edge> works;

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          NodeData& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          if (sdata.component() == c)
            return;
          auto beg = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
          const auto end = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);

          for (std::advance(beg, NEIGHBOR_SAMPLES.getValue()); beg < end;
               beg++) {
            GNode dst = graph.getEdgeDst(beg);
            NodeData& ddata =
                graph.getData(dst, galois::MethodFlag::UNPROTECTED);
            if (src < dst || c == ddata.component()) {
              works.push_back(std::make_pair(src, dst));
            }
          }
        },
        galois::loopname("EdgeAfforest-LCS-Assembling"), galois::steal());

    galois::for_each(
        galois::iterate(works),
        [&](const Edge& e, auto& ctx) {
          NodeData& sdata =
              graph.getData(e.first, galois::MethodFlag::UNPROTECTED);
          if (sdata.component() == c)
            return;
          NodeData& ddata =
              graph.getData(e.second, galois::MethodFlag::UNPROTECTED);
          component_type victim = sdata.hook_min(&ddata, c);
          if (victim) {
            GNode src = victim - c0; // TODO (bozhi) tricky!
            for (auto ii : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
              GNode dst = graph.getEdgeDst(ii);
              ctx.push_back(std::make_pair(dst, src));
            }
          }
        },
        galois::disable_conflict_detection(),
        galois::loopname("EdgeAfforest-LCS-Link"));

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          NodeData& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          sdata.compress();
        },
        galois::steal(), galois::loopname("EdgeAfforest-LCS-Compress"));
  }
};

/**
 * Edgetiled CC w/ Afforest sampling
 */
struct EdgeTiledAfforestAlgo {
  struct NodeData : public galois::UnionFindNode<NodeData> {
    using component_type = NodeData*;

    NodeData() : galois::UnionFindNode<NodeData>(const_cast<NodeData*>(this)) {}
    NodeData(const NodeData& o)
        : galois::UnionFindNode<NodeData>(o.m_component) {}

    component_type component() { return this->get(); }
    bool isRepComp(unsigned int) { return false; } // verify

  public:
    void link(NodeData* b) {
      NodeData* a = m_component.load(std::memory_order_relaxed);
      b           = b->m_component.load(std::memory_order_relaxed);
      while (a != b) {
        if (a < b)
          std::swap(a, b);
        // Now a > b
        NodeData* ac = a->m_component.load(std::memory_order_relaxed);
        if ((ac == a && a->m_component.compare_exchange_strong(a, b)) ||
            (b == ac))
          break;
        a = (a->m_component.load(std::memory_order_relaxed))
                ->m_component.load(std::memory_order_relaxed);
        b = b->m_component.load(std::memory_order_relaxed);
      }
    }
  };
  using Graph =
      galois::graphs::LC_CSR_Graph<NodeData,
                                   void>::with_no_lockable<true>::type;
  using GNode          = Graph::GraphNode;
  using component_type = NodeData::component_type;

  struct EdgeTile {
    GNode src;
    Graph::edge_iterator beg;
    Graph::edge_iterator end;
  };

  template <typename G>
  void readGraph(G& graph) {
    galois::graphs::readGraph(graph, inputFile);
  }

  void operator()(Graph& graph) {
    // (bozhi) should NOT go through single direction in sampling step: nodes
    // with edges less than NEIGHBOR_SAMPLES will fail
    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          auto ii = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
          const auto end = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
          for (uint32_t r = 0; r < NEIGHBOR_SAMPLES && ii < end; ++r, ++ii) {
            GNode dst = graph.getEdgeDst(ii);
            NodeData& sdata =
                graph.getData(src, galois::MethodFlag::UNPROTECTED);
            NodeData& ddata =
                graph.getData(dst, galois::MethodFlag::UNPROTECTED);
            sdata.link(&ddata);
          }
        },
        galois::steal(), galois::loopname("EdgetiledAfforest-VNS-Link"));

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          NodeData& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          sdata.compress();
        },
        galois::steal(), galois::loopname("EdgetiledAfforest-VNS-Compress"));

    galois::StatTimer StatTimer_Sampling("EdgetiledAfforest-LCS-Sampling");
    StatTimer_Sampling.start();
    const component_type c = approxLargestComponent<component_type>(graph);
    StatTimer_Sampling.stop();

    galois::InsertBag<EdgeTile> works;
    std::cout << "INFO: Using edge tile size of " << EDGE_TILE_SIZE
              << " and chunk size of " << CHUNK_SIZE << "\n";
    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          NodeData& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          if (sdata.component() == c)
            return;
          auto beg = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
          const auto end = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);

          for (std::advance(beg, NEIGHBOR_SAMPLES.getValue());
               beg + EDGE_TILE_SIZE < end;) {
            auto ne = beg + EDGE_TILE_SIZE;
            assert(ne < end);
            works.push_back(EdgeTile{src, beg, ne});
            beg = ne;
          }

          if ((end - beg) > 0) {
            works.push_back(EdgeTile{src, beg, end});
          }
        },
        galois::loopname("EdgetiledAfforest-LCS-Tiling"), galois::steal());

    galois::do_all(
        galois::iterate(works),
        [&](const EdgeTile& tile) {
          NodeData& sdata =
              graph.getData(tile.src, galois::MethodFlag::UNPROTECTED);
          if (sdata.component() == c)
            return;
          for (auto ii = tile.beg; ii < tile.end; ++ii) {
            GNode dst = graph.getEdgeDst(ii);
            NodeData& ddata =
                graph.getData(dst, galois::MethodFlag::UNPROTECTED);
            sdata.link(&ddata);
          }
        },
        galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
        galois::loopname("EdgetiledAfforest-LCS-Link"));

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          NodeData& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          sdata.compress();
        },
        galois::steal(), galois::loopname("EdgetiledAfforest-LCS-Compress"));
  }
};

template <typename Graph>
bool verify(
    Graph&,
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
        std::cerr << std::dec << "not in same component: " << (unsigned int)n
                  << " (" << me.component() << ")"
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

  using Map = galois::gstl::Map<component_type, int>;

  auto reduce = [](Map& lhs, Map&& rhs) -> Map& {
    Map v{std::move(rhs)};

    for (auto& kv : v) {
      if (lhs.count(kv.first) == 0) {
        lhs[kv.first] = 0;
      }
      lhs[kv.first] += kv.second;
    }

    return lhs;
  };

  auto mapIdentity = []() { return Map(); };

  auto accumMap = galois::make_reducible(reduce, mapIdentity);

  galois::GAccumulator<size_t> accumReps;

  galois::do_all(
      galois::iterate(graph),
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
        accumMap.update(Map{std::make_pair(n.component(), 1)});
      },
      galois::loopname("CountLargest"));

  Map& map    = accumMap.reduce();
  size_t reps = accumReps.reduce();

  using ComponentSizePair = std::pair<component_type, int>;

  auto sizeMax = [](const ComponentSizePair& a, const ComponentSizePair& b) {
    if (a.second > b.second) {
      return a;
    }
    return b;
  };

  auto identity = []() { return ComponentSizePair{}; };

  auto maxComp = galois::make_reducible(sizeMax, identity);

  galois::do_all(galois::iterate(map),
                 [&](const ComponentSizePair& x) { maxComp.update(x); });

  ComponentSizePair largest = maxComp.reduce();

  // Compensate for dropping representative node of components
  double ratio       = graph.size() - reps + map.size();
  size_t largestSize = largest.second + 1;
  if (ratio) {
    ratio = largestSize / ratio;
  }

  std::cout << "Total components: " << reps << "\n";
  std::cout << "Number of non-trivial components: " << map.size()
            << " (largest size: " << largestSize << " [" << ratio << "])\n";

  return largest.first;
}

template <typename Graph>
void initialize(Graph&) {}

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

  galois::StatTimer execTime("Timer_0");
  execTime.start();
  algo(graph);
  execTime.stop();

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
  LonestarStart(argc, argv, name, desc, nullptr, &inputFile);

  galois::StatTimer totalTime("TimerTotal");
  totalTime.start();

  if (!symmetricGraph) {
    GALOIS_DIE("This application requires a symmetric graph input;"
               " please use the -symmetricGraph flag "
               " to indicate the input is a symmetric graph.");
  }

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
  case Algo::afforest:
    run<AfforestAlgo>();
    break;
  case Algo::edgeafforest:
    run<EdgeAfforestAlgo>();
    break;
  case Algo::edgetiledafforest:
    run<EdgeTiledAfforestAlgo>();
    break;

  default:
    std::cerr << "Unknown algorithm\n";
    abort();
  }

  totalTime.stop();

  return 0;
}
