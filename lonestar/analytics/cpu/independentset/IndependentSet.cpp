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
#include "galois/Bag.h"
#include "galois/ParallelSTL.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/runtime/Profile.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <random>
#include <math.h>

const char* name = "Maximal Independent Set";
const char* desc =
    "Computes a maximal independent set (not maximum) of nodes in a graph";
const char* url = "independent_set";

enum Algo { serial, pull, nondet, detBase, prio, edgetiledprio };

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);

static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm:"),
    cll::values(
        clEnumVal(serial, "Serial"),
        clEnumVal(pull,
                  "Pull-based (node 0 is initially in the independent set)"),
        clEnumVal(nondet, "Non-deterministic, use bulk synchronous worklist"),
        clEnumVal(detBase, "use deterministic worklist"),
        clEnumVal(
            prio,
            "prio algo based on Martin's GPU ECL-MIS algorithm (default)"),
        clEnumVal(
            edgetiledprio,
            "edge-tiled prio algo based on Martin's GPU ECL-MIS algorithm")),
    cll::init(prio));

enum MatchFlag : char { UNMATCHED, OTHER_MATCHED, MATCHED };

struct Node {
  MatchFlag flag;
  Node() : flag(UNMATCHED) {}
};

struct prioNode {
  unsigned char flag; // 1 bit matched,6 bits prio, 1 bit undecided
  prioNode() : flag((unsigned char){0x01}) {}
};

struct SerialAlgo {
  using Graph = galois::graphs::LC_CSR_Graph<Node, void>::with_numa_alloc<
      true>::type ::with_no_lockable<true>::type;
  using GNode = Graph::GraphNode;

  void operator()(Graph& graph) {
    for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
      if (findUnmatched(graph, *ii))
        match(graph, *ii);
    }
  }

  bool findUnmatched(Graph& graph, GNode src) {
    Node& me = graph.getData(src);
    if (me.flag != UNMATCHED)
      return false;

    for (auto ii : graph.edges(src)) {
      GNode dst  = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst);
      if (data.flag == MATCHED)
        return false;
    }
    return true;
  }

  void match(Graph& graph, GNode src) {
    Node& me = graph.getData(src);
    for (auto ii : graph.edges(src)) {
      GNode dst  = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst);
      data.flag  = OTHER_MATCHED;
    }
    me.flag = MATCHED;
  }
};

template <Algo algo>
struct DefaultAlgo {

  using Graph = typename galois::graphs::LC_CSR_Graph<
      Node, void>::template with_numa_alloc<true>::type;

  using GNode = typename Graph::GraphNode;

  struct LocalState {
    bool mod;
    explicit LocalState() : mod(false) {}
  };

  template <galois::MethodFlag Flag>
  bool build(Graph& graph, GNode src) {
    Node& me = graph.getData(src, Flag);
    if (me.flag != UNMATCHED)
      return false;

    for (auto ii : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
      GNode dst  = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst, Flag);
      if (data.flag == MATCHED)
        return false;
    }
    return true;
  }

  void modify(Graph& graph, GNode src) {
    Node& me = graph.getData(src, galois::MethodFlag::UNPROTECTED);
    for (auto ii : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
      GNode dst  = graph.getEdgeDst(ii);
      Node& data = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
      data.flag  = OTHER_MATCHED;
    }
    me.flag = MATCHED;
  }

  template <typename C>
  void processNode(Graph& graph, const GNode& src, C& ctx) {
    bool mod;
    mod = build<galois::MethodFlag::WRITE>(graph, src);
    graph.getData(src, galois::MethodFlag::WRITE);
    ctx.cautiousPoint(); // Failsafe point

    if (mod) {
      modify(graph, src);
    }
  }

  template <typename WL, typename... Args>
  void run(Graph& graph, Args&&... args) {

    auto detID = [](const GNode& x) { return x; };

    galois::for_each(
        galois::iterate(graph),
        [&, this](const GNode& src, auto& ctx) {
          this->processNode(graph, src, ctx);
        },
        galois::no_pushes(), galois::wl<WL>(), galois::loopname("DefaultAlgo"),
        galois::det_id<decltype(detID)>(detID),
        galois::local_state<LocalState>(), std::forward<Args>(args)...);
  }

  void operator()(Graph& graph) {
    using DWL = galois::worklists::Deterministic<>;

    using BSWL = galois::worklists::BulkSynchronous<
        typename galois::worklists::PerSocketChunkFIFO<64>>;

    switch (algo) {
    case nondet:
      run<BSWL>(graph);
      break;
    case detBase:
      run<DWL>(graph);
      break;
    default:
      std::cerr << "Unknown algorithm" << algo << "\n";
      abort();
    }
  }
};

struct PullAlgo {

  using Graph = galois::graphs::LC_CSR_Graph<Node, void>::with_numa_alloc<
      true>::type ::with_no_lockable<true>::type;

  using GNode = Graph::GraphNode;
  using Bag   = galois::InsertBag<GNode>;

  using Counter = galois::GAccumulator<size_t>;

  template <typename R>
  void pull(const R& range, Graph& graph, Bag& matched, Bag& otherMatched,
            Bag& next, Counter& numProcessed) {

    galois::do_all(
        range,
        [&](const GNode& src) {
          numProcessed += 1;
          Node& n = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          if (n.flag == OTHER_MATCHED)
            return;

          MatchFlag f = MATCHED;
          for (auto edge :
               graph.out_edges(src, galois::MethodFlag::UNPROTECTED)) {
            GNode dst = graph.getEdgeDst(edge);
            if (dst >= src) {
              continue;
            }

            Node& other = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
            if (other.flag == MATCHED) {
              f = OTHER_MATCHED;
              break;
            } else if (other.flag == UNMATCHED) {
              f = UNMATCHED;
            }
          }

          if (f == UNMATCHED) {
            next.push_back(src);
          } else if (f == MATCHED) {
            matched.push_back(src);
          } else {
            otherMatched.push_back(src);
          }
        },
        galois::loopname("pull"));
  }

  template <MatchFlag F>
  void take(Bag& bag, Graph& graph, Counter& numTaken) {

    galois::do_all(
        galois::iterate(bag),
        [&](const GNode& src) {
          Node& n = graph.getData(src, galois::MethodFlag::UNPROTECTED);
          numTaken += 1;
          n.flag = F;
        },
        galois::loopname("take"));
  }

  void operator()(Graph& graph) {
    size_t rounds = 0;
    Counter numProcessed;
    Counter numTaken;

    Bag bags[2];
    Bag* cur  = &bags[0];
    Bag* next = &bags[1];
    Bag matched;
    Bag otherMatched;
    uint64_t size  = graph.size();
    uint64_t delta = graph.size() / 25;

    Graph::iterator ii = graph.begin();
    Graph::iterator ei = graph.begin();

    while (size > 0) {
      numProcessed.reset();

      if (!cur->empty()) {
        pull(galois::iterate(*cur), graph, matched, otherMatched, *next,
             numProcessed);
      }

      size_t numCur = numProcessed.reduce();
      std::advance(ei, std::min(size, delta) - numCur);

      if (ii != ei) {
        pull(galois::iterate(ii, ei), graph, matched, otherMatched, *next,
             numProcessed);
      }

      ii = ei;

      numTaken.reset();

      take<MATCHED>(matched, graph, numTaken);
      take<OTHER_MATCHED>(otherMatched, graph, numTaken);

      cur->clear();
      matched.clear();
      otherMatched.clear();
      std::swap(cur, next);
      rounds += 1;
      assert(size >= numTaken.reduce());
      size -= numTaken.reduce();
    }

    galois::runtime::reportStat_Single("IndependentSet-PullAlgo", "rounds",
                                       rounds);
  }
};

struct PrioAlgo {
  using Graph = galois::graphs::LC_CSR_Graph<prioNode, void>::with_numa_alloc<
      true>::type ::with_no_lockable<true>::type;
  using GNode = Graph::GraphNode;

  unsigned int hash(unsigned int val) const {
    val = ((val >> 16) ^ val) * 0x45d9f3b;
    val = ((val >> 16) ^ val) * 0x45d9f3b;
    return (val >> 16) ^ val;
  }

  void operator()(Graph& graph) {
    galois::GAccumulator<size_t> rounds;
    galois::GAccumulator<float> nedges;
    galois::GReduceLogicalOr unmatched;
    galois::substrate::PerThreadStorage<std::mt19937*> generator;

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          nedges += std::distance(
              graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
              graph.edge_end(src, galois::MethodFlag::UNPROTECTED));
        },
        galois::loopname("cal_degree"), galois::steal());

    float nedges_tmp = nedges.reduce();
    float avg_degree = nedges_tmp / (float)graph.size();
    unsigned char in = ~1;
    float scale_avg  = ((in / 2) - 1) * avg_degree;

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          prioNode& nodedata =
              graph.getData(src, galois::MethodFlag::UNPROTECTED);
          float degree = (float)std::distance(
              graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
              graph.edge_end(src, galois::MethodFlag::UNPROTECTED));
          float x = degree - hash(src) * 0.00000000023283064365386962890625f;
          int res = round(scale_avg / (avg_degree + x));
          unsigned char val = (res + res) | 1;
          nodedata.flag     = val;
        },
        galois::loopname("init-prio"), galois::steal());

    do {
      unmatched.reset();
      galois::do_all(
          galois::iterate(graph),
          [&](const GNode& src) {
            prioNode& nodedata =
                graph.getData(src, galois::MethodFlag::UNPROTECTED);

            if (!(nodedata.flag & (unsigned char)1))
              return;

            for (auto edge :
                 graph.out_edges(src, galois::MethodFlag::UNPROTECTED)) {
              GNode dst = graph.getEdgeDst(edge);

              prioNode& other =
                  graph.getData(dst, galois::MethodFlag::UNPROTECTED);

              if (other.flag == (unsigned char)0xfe) { // matched, highest prio
                nodedata.flag = (unsigned char)0x00;
                unmatched.update(true);
                return;
              }

              if (nodedata.flag > other.flag)
                continue;
              else if (nodedata.flag == other.flag) {
                if (src > dst)
                  continue;
                else if (src == dst) {
                  nodedata.flag = (unsigned char)0x00; // other_matched
                  return;
                } else {
                  unmatched.update(true);
                  return;
                }
              } else {
                unmatched.update(true);
                return;
              }
            }
            nodedata.flag = (unsigned char)0xfe; // matched, highest prio
          },
          galois::loopname("execute"), galois::steal());

      rounds += 1;
    } while (unmatched.reduce());

    galois::runtime::reportStat_Single("IndependentSet-prioAlgo", "rounds",
                                       rounds.reduce());
  }
};

struct EdgeTiledPrioAlgo {
  using Graph = galois::graphs::LC_CSR_Graph<prioNode, void>::with_numa_alloc<
      true>::type ::with_no_lockable<true>::type;
  using GNode = Graph::GraphNode;

  struct EdgeTile {
    GNode src;
    Graph::edge_iterator beg;
    Graph::edge_iterator end;
    bool flag;
  };

  unsigned int hash(unsigned int val) const {
    val = ((val >> 16) ^ val) * 0x45d9f3b;
    val = ((val >> 16) ^ val) * 0x45d9f3b;
    return (val >> 16) ^ val;
  }

  void operator()(Graph& graph) {
    galois::GAccumulator<size_t> rounds;
    galois::GAccumulator<float> nedges;
    galois::GReduceLogicalOr unmatched;
    galois::substrate::PerThreadStorage<std::mt19937*> generator;
    galois::InsertBag<EdgeTile> works;
    const int EDGE_TILE_SIZE = 64;
    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          nedges += std::distance(
              graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
              graph.edge_end(src, galois::MethodFlag::UNPROTECTED));
        },
        galois::loopname("cal_degree"), galois::steal());

    float nedges_tmp = nedges.reduce();
    float avg_degree = nedges_tmp / (float)graph.size();
    unsigned char in = ~1;
    float scale_avg  = ((in / 2) - 1) * avg_degree;

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          prioNode& nodedata =
              graph.getData(src, galois::MethodFlag::UNPROTECTED);
          auto beg = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED);
          const auto end = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);

          float degree = (float)std::distance(beg, end);
          float x = degree - hash(src) * 0.00000000023283064365386962890625f;
          int res = round(scale_avg / (avg_degree + x));
          unsigned char val = (res + res) | 0x03;

          nodedata.flag = val;
          assert(beg <= end);
          if ((end - beg) > EDGE_TILE_SIZE) {
            for (; beg + EDGE_TILE_SIZE < end;) {
              auto ne = beg + EDGE_TILE_SIZE;
              assert(ne < end);
              works.push_back(EdgeTile{src, beg, ne, false});
              beg = ne;
            }
          }
          if ((end - beg) > 0) {
            works.push_back(EdgeTile{src, beg, end, false});
          }
        },
        galois::loopname("init-prio"), galois::steal());

    do {
      unmatched.reset();
      galois::do_all(
          galois::iterate(works),
          [&](EdgeTile& tile) {
            GNode src = tile.src;

            prioNode& nodedata =
                graph.getData(src, galois::MethodFlag::UNPROTECTED);

            if ((nodedata.flag & (unsigned char){1})) { // is undecided

              for (auto edge = tile.beg; edge != tile.end; ++edge) {
                GNode dst = graph.getEdgeDst(edge);

                prioNode& other =
                    graph.getData(dst, galois::MethodFlag::UNPROTECTED);

                if (other.flag ==
                    (unsigned char){0xfe}) { // permanent matched, highest prio
                  nodedata.flag = (unsigned char){0x00};
                  return;
                }

                if (nodedata.flag > other.flag)
                  continue;
                else if (nodedata.flag == other.flag) {
                  if (src > dst)
                    continue;
                  else if (src == dst) {
                    nodedata.flag = (unsigned char){0x00}; // other_matched
                    tile.flag     = false;
                    return;
                  } else {
                    tile.flag = false;
                    unmatched.update(true);
                    return;
                  }
                } else {
                  tile.flag = false;
                  unmatched.update(true);
                  return;
                }
              }
              tile.flag = true; // temporary-matched
            }
          },
          galois::loopname("execute"), galois::steal());

      galois::do_all(
          galois::iterate(works),
          [&](EdgeTile& tile) {
            auto src = tile.src;
            prioNode& nodedata =
                graph.getData(src, galois::MethodFlag::UNPROTECTED);

            if ((nodedata.flag & (unsigned char){1}) &&
                tile.flag == false) { // undecided and temporary no
              nodedata.flag &=
                  (unsigned char){0xfd}; // 0x1111 1101, not temporary yes
            }
          },
          galois::loopname("match_reduce"), galois::steal());

      galois::do_all(
          galois::iterate(graph),
          [&](const GNode& src) {
            prioNode& nodedata =
                graph.getData(src, galois::MethodFlag::UNPROTECTED);
            if ((nodedata.flag & (unsigned char){0x01}) != 0) { // undecided
              if ((nodedata.flag & (unsigned char){0x02}) !=
                  0) { // temporary yes
                nodedata.flag =
                    (unsigned char){0xfe}; // 0x1111 1110, permanent yes
                for (auto edge :
                     graph.out_edges(src, galois::MethodFlag::UNPROTECTED)) {
                  GNode dst = graph.getEdgeDst(edge);

                  prioNode& other =
                      graph.getData(dst, galois::MethodFlag::UNPROTECTED);
                  other.flag =
                      (unsigned char){0x00}; // OTHER_MATCHED, permanent no
                }
              } else
                nodedata.flag |=
                    (unsigned char){0x03}; // 0x0000 0011, temp yes, undecided
            }
          },
          galois::loopname("match_update"), galois::steal());

      rounds += 1;
    } while (unmatched.reduce());

    galois::runtime::reportStat_Single("IndependentSet-prioAlgo", "rounds",
                                       rounds.reduce());
  }
};

template <typename Graph>
struct is_bad {
  using GNode = typename Graph::GraphNode;
  using Node  = typename Graph::node_data_type;
  Graph& graph;

  is_bad(Graph& g) : graph(g) {}

  bool operator()(GNode n) const {
    Node& me = graph.getData(n);
    if (me.flag == MATCHED) {
      for (auto ii : graph.edges(n)) {
        GNode dst  = graph.getEdgeDst(ii);
        Node& data = graph.getData(dst);
        if (dst != n && data.flag == MATCHED) {
          std::cerr << "double match\n";
          return true;
        }
      }
    } else if (me.flag == UNMATCHED) {
      bool ok = false;
      for (auto ii : graph.edges(n)) {
        GNode dst  = graph.getEdgeDst(ii);
        Node& data = graph.getData(dst);
        if (data.flag != UNMATCHED) {
          ok = true;
        }
      }
      if (!ok) {
        std::cerr << "not maximal\n";
        return true;
      }
    }
    return false;
  }
};

template <typename Graph>
struct is_matched {
  Graph& graph;
  using GNode = typename Graph::GraphNode;

  is_matched(Graph& g) : graph(g) {}

  bool operator()(const GNode& n) const {
    return graph.getData(n).flag == MATCHED;
  }
};

template <typename Graph, typename Algo>
bool verify(Graph& graph, Algo&) {
  using GNode    = typename Graph::GraphNode;
  using prioNode = typename Graph::node_data_type;

  if (std::is_same<Algo, PrioAlgo>::value ||
      std::is_same<Algo, EdgeTiledPrioAlgo>::value) {
    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          prioNode& nodedata =
              graph.getData(src, galois::MethodFlag::UNPROTECTED);
          if (nodedata.flag == (unsigned char){0xfe}) {
            nodedata.flag = MATCHED;
          } else if (nodedata.flag == (unsigned char){0x00}) {
            nodedata.flag = OTHER_MATCHED;
          } else
            std::cout << "error in verify_change! Some nodes are not decided."
                      << "\n";
        },
        galois::loopname("verify_change"));
  }

  return galois::ParallelSTL::find_if(graph.begin(), graph.end(),
                                      is_bad<Graph>(graph)) == graph.end();
}

template <typename Algo>
void run() {
  using Graph = typename Algo::Graph;
  using GNode = typename Graph::GraphNode;

  Algo algo;
  Graph graph;
  galois::graphs::readGraph(graph, inputFile);

  // galois::preAlloc(numThreads + (graph.size() * sizeof(Node) * numThreads /
  // 8) / galois::runtime::MM::hugePageSize); Tighter upper bound
  if (std::is_same<Algo, DefaultAlgo<nondet>>::value) {
    galois::preAlloc(numThreads +
                     16 * graph.size() / galois::runtime::pagePoolSize());
  } else {
    galois::preAlloc(numThreads + 64 * (sizeof(GNode) + sizeof(Node)) *
                                      graph.size() /
                                      galois::runtime::pagePoolSize());
  }

  galois::reportPageAlloc("MeminfoPre");
  galois::StatTimer execTime("Timer_0");

  execTime.start();
  algo(graph);
  execTime.stop();

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify && !verify(graph, algo)) {
    std::cerr << "verification failed\n";
    assert(0 && "verification failed");
    abort();
  }

  std::cout << "Cardinality of maximal independent set: "
            << galois::ParallelSTL::count_if(graph.begin(), graph.end(),
                                             is_matched<Graph>(graph))
            << "\n";
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url, &inputFile);

  galois::StatTimer totalTime("TimerTotal");
  totalTime.start();

  if (!symmetricGraph) {
    GALOIS_DIE("independent set requires a symmetric graph input;"
               " please use the -symmetricGraph flag "
               " to indicate the input is a symmetric graph");
  }

  if (!symmetricGraph) {
    GALOIS_DIE("This application requires a symmetric graph input;"
               " please use the -symmetricGraph flag "
               " to indicate the input is a symmetric graph.");
  }

  switch (algo) {
  case serial:
    run<SerialAlgo>();
    break;
  case nondet:
    run<DefaultAlgo<nondet>>();
    break;
  case detBase:
    run<DefaultAlgo<detBase>>();
    break;
  case pull:
    run<PullAlgo>();
    break;
  case prio:
    run<PrioAlgo>();
    break;
  case edgetiledprio:
    run<EdgeTiledPrioAlgo>();
    break;
  default:
    std::cerr << "Unknown algorithm" << algo << "\n";
    abort();
  }

  totalTime.stop();

  return 0;
}
