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
#include "galois/gstl.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include "Lonestar/BFS_SSSP.h"

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
    filename(cll::Positional, cll::desc("<input graph>"), cll::Required);

static cll::opt<unsigned int>
    startNode("startNode",
              cll::desc("Node to start search from (default value 0)"),
              cll::init(0));
static cll::opt<unsigned int>
    reportNode("reportNode",
               cll::desc("Node to report distance to (default value 1)"),
               cll::init(1));
// static cll::opt<unsigned int> stepShiftw("delta",
// cll::desc("Shift value for the deltastep"),
// cll::init(10));

enum Exec { SERIAL, PARALLEL };

enum Algo { AsyncTile = 0, Async, SyncTile, Sync, Sync2pTile, Sync2p };

const char* const ALGO_NAMES[] = {"AsyncTile", "Async",      "SyncTile",
                                  "Sync",      "Sync2pTile", "Sync2p"};

static cll::opt<Exec> execution(
    "exec",
    cll::desc("Choose SERIAL or PARALLEL execution (default value PARALLEL):"),
    cll::values(clEnumVal(SERIAL, "SERIAL"), clEnumVal(PARALLEL, "PARALLEL"),
                clEnumValEnd),
    cll::init(PARALLEL));

static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm (default value SyncTile):"),
    cll::values(clEnumVal(AsyncTile, "AsyncTile"), clEnumVal(Async, "Async"),
                clEnumVal(SyncTile, "SyncTile"), clEnumVal(Sync, "Sync"),
                clEnumVal(Sync2pTile, "Sync2pTile"),
                clEnumVal(Sync2p, "Sync2p"), clEnumValEnd),
    cll::init(SyncTile));

using Graph =
    galois::graphs::LC_CSR_Graph<unsigned, void>::with_no_lockable<true>::type;
//::with_numa_alloc<true>::type;

using GNode = Graph::GraphNode;

constexpr static const bool TRACK_WORK          = false;
constexpr static const unsigned CHUNK_SIZE      = 256u;
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
  void operator()(C& cont, const GNode& n, const char* const _parallel) const {
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
  void operator()(C& cont, const GNode& n, const char* const _parallel) const {
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
  void operator()(C& cont, const GNode& n, const char* const _parallel) const {
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

  constexpr bool useCAS = CONCURRENT && !std::is_same<WL, BSWL>::value;

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

  loop(galois::iterate(initBag),
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
       galois::wl<WL>(), galois::loopname("runBFS"), galois::no_conflicts());

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

  Cont* curr = new Cont();
  Cont* next = new Cont();

  Dist nextLevel              = 0u;
  graph.getData(source, flag) = 0u;

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

    loop(galois::iterate(*curr),
         [&](const T& item) {
           for (auto e : edgeRange(item)) {
             auto dst = graph.getEdgeDst(e);
             // if(dst == 13 || dst == 2 || dst == 51) std::cout<<" node " <<
             // dst << " visited"<<std::endl;
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

  delete curr;
  delete next;
}

template <bool CONCURRENT, typename P, typename R>
void sync2phaseAlgo(Graph& graph, GNode source, const P& pushWrap,
                    const R& edgeRange) {

  using NodeCont =
      typename std::conditional<CONCURRENT, galois::InsertBag<GNode>,
                                galois::SerStack<GNode>>::type;
  using TileCont =
      typename std::conditional<CONCURRENT, galois::InsertBag<EdgeTile>,
                                galois::SerStack<EdgeTile>>::type;

  using Loop = typename std::conditional<CONCURRENT, galois::DoAll,
                                         galois::StdForEach>::type;

  constexpr galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;

  Loop loop;

  NodeCont activeNodes;
  TileCont edgeTiles;

  Dist nextLevel              = 0u;
  graph.getData(source, flag) = 0u;

  activeNodes.push(source);

  while (!activeNodes.empty()) {

    loop(galois::iterate(activeNodes),
         [&](const GNode& src) {
           pushWrap(edgeTiles, src);
         },
         galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
         galois::loopname("activeNodes"));

    ++nextLevel;
    activeNodes.clear();

    loop(galois::iterate(edgeTiles),
         [&](const EdgeTile& item) {
           for (auto e : edgeRange(item)) {
             auto dst      = graph.getEdgeDst(e);
             auto& dstData = graph.getData(dst, flag);

             if (dstData == BFS::DIST_INFINITY) {
               dstData = nextLevel;
               activeNodes.push(dst);
             }
           }
         },
         galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
         galois::loopname("edgeTiles"));

    edgeTiles.clear();
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
  case Sync2pTile:
    sync2phaseAlgo<CONCURRENT>(graph, source, EdgeTilePushWrap{graph},
                               TileRangeFn());
    break;
  case Sync2p:
    sync2phaseAlgo<CONCURRENT>(graph, source, OneTilePushWrap{graph},
                               TileRangeFn());
    break;
  default:
    std::cerr << "ERROR: unkown algo type" << std::endl;
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  Graph graph;
  GNode source, report;

  std::cout << "Reading from file: " << filename << std::endl;
  galois::graphs::readGraph(graph, filename);
  std::cout << "Read " << graph.size() << " nodes, " << graph.sizeEdges()
            << " edges" << std::endl;

  if (startNode >= graph.size() || reportNode >= graph.size()) {
    std::cerr << "failed to set report: " << reportNode
              << " or failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }

  auto it = graph.begin();
  std::advance(it, startNode);
  source = *it;
  it     = graph.begin();
  std::advance(it, reportNode);
  report = *it;

  size_t approxNodeData = 4 * (graph.size() + graph.sizeEdges());
  // size_t approxEdgeData = graph.sizeEdges() * sizeof(typename
  // Graph::edge_data_type) * 2;
  galois::preAlloc(8 * numThreads +
                   approxNodeData / galois::runtime::pagePoolSize());

  galois::reportPageAlloc("MeminfoPre");

  galois::do_all(galois::iterate(graph),
                 [&graph](GNode n) { graph.getData(n) = BFS::DIST_INFINITY; });
  graph.getData(source) = 0;

  std::cout << "Running " << ALGO_NAMES[algo] << " algorithm with "
            << (bool(execution) ? "PARALLEL" : "SERIAL") << " execution "
            << std::endl;

  galois::StatTimer Tmain;
  Tmain.start();

  if (execution == SERIAL) {
    runAlgo<false>(graph, source);
  } else if (execution == PARALLEL) {
    runAlgo<true>(graph, source);
  } else {
    std::cerr << "ERROR: unknown type of execution passed to -exec"
              << std::endl;
    std::abort();
  }

  Tmain.stop();

  galois::reportPageAlloc("MeminfoPost");

  std::cout << "Node " << reportNode << " has distance "
            << graph.getData(report) << "\n";

  if (!skipVerify) {
    if (BFS::verify(graph, source)) {
      std::cout << "Verification successful.\n";
    } else {
      GALOIS_DIE("Verification failed");
    }
  }

  return 0;
}
