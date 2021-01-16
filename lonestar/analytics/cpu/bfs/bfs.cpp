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
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "Lonestar/BoilerPlate.h"
#include "Lonestar/BFS_SSSP.h"

#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <deque>
#include <type_traits>

namespace cll = llvm::cl;

static const char* name = "Breadth-first Search";

static const char* desc =
    "Computes the shortest path from a source node to all nodes in a directed "
    "graph using a modified Bellman-Ford algorithm";

static const char* url = "breadth_first_search";

static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> startNodesFile(
    "startNodesFile",
    cll::desc("File containing whitespace separated list of source "
              "nodes for computing betweenness centrality; "
              "if set, -startNode is ignored"));
static cll::opt<std::string> startNodesString(
    "startNodes",
    cll::desc("String containing whitespace separated list of source nodes for "
              "computing betweenness centrality (default value '0'); ignore if "
              "-startNodesFile is used"),
    cll::init("0"));
static cll::opt<unsigned int>
    reportNode("reportNode",
               cll::desc("Node to report distance to (default value 1)"),
               cll::init(1));

// static cll::opt<unsigned int> stepShiftw("delta",
// cll::desc("Shift value for the deltastep"),
// cll::init(10));

enum Exec { SERIAL, PARALLEL };

enum Algo { AsyncTile = 0, Async, SyncTile, Sync };

const char* const ALGO_NAMES[] = {"AsyncTile", "Async", "SyncTile", "Sync"};

static cll::opt<Exec> execution(
    "exec",
    cll::desc("Choose SERIAL or PARALLEL execution (default value PARALLEL):"),
    cll::values(clEnumVal(SERIAL, "SERIAL"), clEnumVal(PARALLEL, "PARALLEL")),
    cll::init(PARALLEL));

static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm (default value SyncTile):"),
    cll::values(clEnumVal(AsyncTile, "AsyncTile"), clEnumVal(Async, "Async"),
                clEnumVal(SyncTile, "SyncTile"), clEnumVal(Sync, "Sync")),
    cll::init(SyncTile));

using Graph =
    galois::graphs::LC_CSR_Graph<unsigned, void>::with_no_lockable<true>::type;
//::with_numa_alloc<true>::type;

using GNode = Graph::GraphNode;

constexpr static const bool TRACK_WORK          = false;
constexpr static const unsigned CHUNK_SIZE      = 256U;
constexpr static const ptrdiff_t EDGE_TILE_SIZE = 256;

using BFS = BFS_SSSP<Graph, unsigned int, false, EDGE_TILE_SIZE>;

using UpdateRequest       = BFS::UpdateRequest;
using Dist                = BFS::Dist;
using SrcEdgeTile         = BFS::SrcEdgeTile;
using SrcEdgeTileMaker    = BFS::SrcEdgeTileMaker;
using SrcEdgeTilePushWrap = BFS::SrcEdgeTilePushWrap;
using ReqPushWrap         = BFS::ReqPushWrap;
using OutEdgeRangeFn      = BFS::OutEdgeRangeFn;
using TileRangeFn         = BFS::TileRangeFn;

struct EdgeTile {
  Graph::edge_iterator beg;
  Graph::edge_iterator end;
};

struct EdgeTileMaker {
  EdgeTile operator()(Graph::edge_iterator beg,
                      Graph::edge_iterator end) const {
    return EdgeTile{beg, end};
  }
};

struct NodePushWrap {

  template <typename C>
  void operator()(C& cont, const GNode& n, const char* const) const {
    (*this)(cont, n);
  }

  template <typename C>
  void operator()(C& cont, const GNode& n) const {
    cont.push(n);
  }
};

struct EdgeTilePushWrap {
  Graph& graph;

  template <typename C>
  void operator()(C& cont, const GNode& n, const char* const) const {
    BFS::pushEdgeTilesParallel(cont, graph, n, EdgeTileMaker{});
  }

  template <typename C>
  void operator()(C& cont, const GNode& n) const {
    BFS::pushEdgeTiles(cont, graph, n, EdgeTileMaker{});
  }
};

struct OneTilePushWrap {
  Graph& graph;

  template <typename C>
  void operator()(C& cont, const GNode& n, const char* const) const {
    (*this)(cont, n);
  }

  template <typename C>
  void operator()(C& cont, const GNode& n) const {
    EdgeTile t{graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
               graph.edge_end(n, galois::MethodFlag::UNPROTECTED)};

    cont.push(t);
  }
};

template <bool CONCURRENT, typename T, typename P, typename R>
void asyncAlgo(Graph& graph, GNode source, const P& pushWrap,
               const R& edgeRange) {

  namespace gwl = galois::worklists;
  // typedef PerSocketChunkFIFO<CHUNK_SIZE> dFIFO;
  using FIFO = gwl::PerSocketChunkFIFO<CHUNK_SIZE>;
  using BSWL = gwl::BulkSynchronous<gwl::PerSocketChunkLIFO<CHUNK_SIZE>>;
  using WL   = FIFO;

  using Loop =
      typename std::conditional<CONCURRENT, galois::ForEach,
                                galois::WhileQ<galois::SerFIFO<T>>>::type;

  GALOIS_GCC7_IGNORE_UNUSED_BUT_SET
  constexpr bool useCAS = CONCURRENT && !std::is_same<WL, BSWL>::value;
  GALOIS_END_GCC7_IGNORE_UNUSED_BUT_SET

  Loop loop;

  galois::GAccumulator<size_t> BadWork;
  galois::GAccumulator<size_t> WLEmptyWork;

  graph.getData(source) = 0;
  galois::InsertBag<T> initBag;

  if (CONCURRENT) {
    pushWrap(initBag, source, 1, "parallel");
  } else {
    pushWrap(initBag, source, 1);
  }

  loop(
      galois::iterate(initBag),
      [&](const T& item, auto& ctx) {
        constexpr galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;

        const auto& sdist = graph.getData(item.src, flag);

        if (TRACK_WORK) {
          if (item.dist != sdist) {
            WLEmptyWork += 1;
            return;
          }
        }

        const auto newDist = item.dist;

        for (auto ii : edgeRange(item)) {
          GNode dst   = graph.getEdgeDst(ii);
          auto& ddata = graph.getData(dst, flag);

          while (true) {

            Dist oldDist = ddata;

            if (oldDist <= newDist) {
              break;
            }

            if (!useCAS ||
                __sync_bool_compare_and_swap(&ddata, oldDist, newDist)) {

              if (!useCAS) {
                ddata = newDist;
              }

              if (TRACK_WORK) {
                if (oldDist != BFS::DIST_INFINITY) {
                  BadWork += 1;
                }
              }

              pushWrap(ctx, dst, newDist + 1);
              break;
            }
          }
        }
      },
      galois::wl<WL>(), galois::loopname("runBFS"),
      galois::disable_conflict_detection());

  if (TRACK_WORK) {
    galois::runtime::reportStat_Single("BFS", "BadWork", BadWork.reduce());
    galois::runtime::reportStat_Single("BFS", "EmptyWork",
                                       WLEmptyWork.reduce());
  }
}

template <bool CONCURRENT, typename T, typename P, typename R>
void syncAlgo(Graph& graph, GNode source, const P& pushWrap,
              const R& edgeRange) {

  using Cont = typename std::conditional<CONCURRENT, galois::InsertBag<T>,
                                         galois::SerStack<T>>::type;
  using Loop = typename std::conditional<CONCURRENT, galois::DoAll,
                                         galois::StdForEach>::type;

  constexpr galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;

  Loop loop;

  auto curr = std::make_unique<Cont>();
  auto next = std::make_unique<Cont>();

  Dist nextLevel              = 0U;
  graph.getData(source, flag) = 0U;

  if (CONCURRENT) {
    pushWrap(*next, source, "parallel");
  } else {
    pushWrap(*next, source);
  }

  assert(!next->empty());

  while (!next->empty()) {

    std::swap(curr, next);
    next->clear();
    ++nextLevel;

    loop(
        galois::iterate(*curr),
        [&](const T& item) {
          for (auto e : edgeRange(item)) {
            auto dst      = graph.getEdgeDst(e);
            auto& dstData = graph.getData(dst, flag);

            if (dstData == BFS::DIST_INFINITY) {
              dstData = nextLevel;
              pushWrap(*next, dst);
            }
          }
        },
        galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
        galois::loopname("Sync"));
  }
}

template <bool CONCURRENT>
void runAlgo(Graph& graph, const GNode& source) {

  switch (algo) {
  case AsyncTile:
    asyncAlgo<CONCURRENT, SrcEdgeTile>(
        graph, source, SrcEdgeTilePushWrap{graph}, TileRangeFn());
    break;
  case Async:
    asyncAlgo<CONCURRENT, UpdateRequest>(graph, source, ReqPushWrap(),
                                         OutEdgeRangeFn{graph});
    break;
  case SyncTile:
    syncAlgo<CONCURRENT, EdgeTile>(graph, source, EdgeTilePushWrap{graph},
                                   TileRangeFn());
    break;
  case Sync:
    syncAlgo<CONCURRENT, GNode>(graph, source, NodePushWrap(),
                                OutEdgeRangeFn{graph});
    break;
  default:
    std::cerr << "ERROR: unkown algo type\n";
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url, &inputFile);

  galois::StatTimer totalTime("TimerTotal");
  galois::StatTimer execTime("Timer_0");
  totalTime.start();

  Graph graph;
  std::cout << "Reading from file: " << inputFile << "\n";
  galois::graphs::readGraph(graph, inputFile);
  std::cout << "Read " << graph.size() << " nodes, " << graph.sizeEdges()
            << " edges\n";

  if (reportNode >= graph.size()) {
    std::cerr << "Failed to set report node: " << reportNode << "\n";
    abort();
  }

  std::vector<uint32_t> startNodes;
  if (!startNodesFile.getValue().empty()) {
    std::ifstream file(startNodesFile);
    if (!file.good()) {
      std::cerr << "Failed to open file: " << startNodesFile << "\n";
      abort();
    }
    startNodes.insert(startNodes.end(), std::istream_iterator<uint64_t>{file},
                      std::istream_iterator<uint64_t>{});
  } else {
    std::istringstream str(startNodesString);
    startNodes.insert(startNodes.end(), std::istream_iterator<uint64_t>{str},
                      std::istream_iterator<uint64_t>{});
  }

  for (auto startNode : startNodes) {
    if (startNode >= graph.size()) {
      std::cerr << "Failed to set start node: " << startNode << "\n";
      abort();
    }
  }

  size_t approxNodeData = 4 * (graph.size() + graph.sizeEdges());
  galois::preAlloc(8 * numThreads +
                   approxNodeData / galois::runtime::pagePoolSize());

  galois::reportPageAlloc("MeminfoPre");

  std::cout << "Running " << ALGO_NAMES[algo] << " algorithm with "
            << (bool(execution) ? "PARALLEL" : "SERIAL") << " execution\n";

  auto it = graph.begin();
  std::advance(it, reportNode.getValue());
  GNode report = *it;

  for (auto startNode : startNodes) {
    it = graph.begin();
    std::advance(it, startNode);
    GNode source = *it;

    galois::do_all(galois::iterate(graph), [&graph](GNode n) {
      graph.getData(n) = BFS::DIST_INFINITY;
    });
    graph.getData(source) = 0;

    execTime.start();

    if (execution == SERIAL) {
      runAlgo<false>(graph, source);
    } else if (execution == PARALLEL) {
      runAlgo<true>(graph, source);
    } else {
      std::cerr << "ERROR: unknown type of execution passed to -exec\n";
    }

    execTime.stop();

    std::cout << "Node " << reportNode << " has distance "
              << graph.getData(report) << " from source " << startNode << "\n";
  }

  galois::reportPageAlloc("MeminfoPost");

  // Sanity checking code
  galois::GReduceMax<uint64_t> maxDistance;
  galois::GAccumulator<uint64_t> distanceSum;
  galois::GAccumulator<uint32_t> visitedNode;
  maxDistance.reset();
  distanceSum.reset();
  visitedNode.reset();

  galois::do_all(
      galois::iterate(graph),
      [&](uint64_t i) {
        uint32_t myDistance = graph.getData(i);

        if (myDistance != BFS::DIST_INFINITY) {
          maxDistance.update(myDistance);
          distanceSum += myDistance;
          visitedNode += 1;
        }
      },
      galois::loopname("Sanity check"), galois::no_stats());

  // report sanity stats
  uint64_t rMaxDistance = maxDistance.reduce();
  uint64_t rDistanceSum = distanceSum.reduce();
  uint64_t rVisitedNode = visitedNode.reduce();
  galois::gInfo("# visited nodes from source ", startNodes.back(), " is ",
                rVisitedNode);
  galois::gInfo("Max distance from source ", startNodes.back(), " is ",
                rMaxDistance);
  galois::gInfo("Sum of visited distances from source ", startNodes.back(),
                " is ", rDistanceSum);

  if (!skipVerify) {
    auto it = graph.begin();
    std::advance(it, startNodes.back());
    GNode source = *it;
    if (BFS::verify(graph, source)) {
      std::cout << "Verification successful.\n";
    } else {
      GALOIS_DIE("verification failed");
    }
  }

  totalTime.stop();

  return 0;
}
