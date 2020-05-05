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

#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"
#include "galois/graphs/B_LC_CSR_Graph.h"
#include "galois/AtomicHelpers.h"
#include "galois/DynamicBitset.h"

#include "Lonestar/BoilerPlate.h"

#include "galois/runtime/Profile.h"

#include "Lonestar/BFS_SSSP.h"

#include <iostream>
#include <deque>
#include <type_traits>

#include <cstdlib>

namespace cll = llvm::cl;

static const char* name = "Breadth-first Search";

static const char* desc =
    "Computes the shortest path from a source node to all nodes in a directed "
    "graph using a modified Bellman-Ford algorithm";

static const char* url = "breadth_first_search";

static cll::opt<std::string>
    filename(cll::Positional, cll::desc("<input graph>"), cll::Required);

static cll::opt<unsigned long long>
    startNode("startNode",
              cll::desc("Node to start search from (default value 0)"),
              cll::init(0));
static cll::opt<unsigned int>
    reportNode("reportNode",
               cll::desc("Node to report distance to (default value 1)"),
               cll::init(1));
static cll::opt<unsigned int>
    numRuns("numRuns",
              cll::desc("Number of runs (default value 1)"),
              cll::init(1));

static cll::opt<int>
    alpha("alpha",
              cll::desc("alpha value to change direction in direction-optimization (default value 15)"),
              cll::init(15));
static cll::opt<int>
    beta("beta",
              cll::desc("beta value to change direction in direction-optimization (default value 18)"),
              cll::init(18));

static cll::opt<unsigned int>
    preAlloc("preAlloc",
              cll::desc("Number of pages to preAlloc (default value 400)"),
              cll::init(400));

static cll::opt<unsigned int>
    numPrint("numPrint",
              cll::desc("Print parents for the numPrint number of nodes for verification if verification is on (default value 10)"),
              cll::init(10));

enum Exec { SERIAL, PARALLEL };

enum Algo { SyncDO = 0, Async};

const char* const ALGO_NAMES[] = {"SyncDO", "Async"};

static cll::opt<Exec> execution(
    "exec",
    cll::desc("Choose SERIAL or PARALLEL execution (default value PARALLEL):"),
    cll::values(clEnumVal(SERIAL, "SERIAL"), clEnumVal(PARALLEL, "PARALLEL")),
    cll::init(PARALLEL));

static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm (default value SyncDO):"),
    cll::values(clEnumVal(SyncDO, "SyncDO"), clEnumVal(Async, "Async")),
    cll::init(SyncDO));

using Graph =
    //galois::graphs::B_LC_CSR_Graph<unsigned, void, false, true, true>;
    galois::graphs::B_LC_CSR_Graph<unsigned, void, false, true, true>;
    //galois::graphs::B_LC_CSR_Graph<unsigned, void>::with_no_lockable<true>::type::with_numa_alloc<true>::type;
using GNode = Graph::GraphNode;


constexpr static const unsigned CHUNK_SIZE      = 256u;
constexpr static const ptrdiff_t EDGE_TILE_SIZE = 256;

using BFS = BFS_SSSP<Graph, unsigned int, false, EDGE_TILE_SIZE>;
using UpdateRequest       = BFS::UpdateRequest;
using Dist                = BFS::Dist;
using OutEdgeRangeFn      = BFS::OutEdgeRangeFn;

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

template <typename WL>
void WlToBitset(WL& wl, galois::DynamicBitSet& bitset) {
  galois::do_all(galois::iterate(wl),
           [&](const GNode& src) {
              bitset.set(src);
           },
           galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
           galois::loopname("WlToBitset"));
}

template <typename WL>
void BitsetToWl(const Graph& graph, const galois::DynamicBitSet& bitset, WL& wl) {
  wl.clear();
  galois::do_all(galois::iterate(graph),
           [&](const GNode& src) {
              if(bitset.test(src))
                //pushWrap(wl, src);
		wl.push(src);
           },
           galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
           galois::loopname("BitsetToWl"));
}

template <bool CONCURRENT, typename T, typename P, typename R>
void syncDOAlgo(Graph& graph, GNode source, const P& pushWrap,
                const R& GALOIS_UNUSED(edgeRange), const uint32_t runID) {

  using Cont = typename std::conditional<CONCURRENT, galois::InsertBag<T>,
                                         galois::SerStack<T>>::type;
  using Loop = typename std::conditional<CONCURRENT, galois::DoAll,
                                         galois::StdForEach>::type;

  constexpr galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
  galois::GAccumulator<uint32_t> work_items;

  Loop loop;

  galois::DynamicBitSet front_bitset,next_bitset;
  front_bitset.resize(graph.size());
  next_bitset.resize(graph.size());

  front_bitset.reset();
  next_bitset.reset();

  Cont* curr = new Cont();
  Cont* next = new Cont();

  Dist nextLevel              = 0u;
  graph.getData(source, flag) = 0u;

  if (CONCURRENT) {
    pushWrap(*next, source, "parallel");
  } else {
    pushWrap(*next, source);
  }
  //adding source to the worklist
  work_items += 1;
  //next_bitset.set(source);

  int64_t edges_to_check = graph.sizeEdges();
  int64_t scout_count = std::distance(graph.edge_begin(source), graph.edge_end(source));
  galois::gPrint("source: ", source, " has OutDegree:", scout_count, "\n");
  assert(!next->empty());

  uint64_t old_workItemNum = 0;
  uint64_t numNodes = graph.size();
  //uint32_t c_pull = 0, c_push = 0;
  galois::GAccumulator<uint64_t> writes_pull, writes_push;
  writes_push.reset();
  writes_pull.reset();
  //std::vector<uint32_t> pull_levels;
  //pull_levels.reserve(10);

  while (!next->empty()) {

    std::swap(curr, next);
    next->clear();
    if (scout_count > edges_to_check / alpha) {

      WlToBitset(*curr, front_bitset);
      do {
        //c_pull++;
        //pull_levels.push_back(nextLevel);

        ++nextLevel;
        old_workItemNum = work_items.reduce();
        work_items.reset();

        //PULL from in-edges
        loop(galois::iterate(graph),
             [&](const T& dst) {
             auto& ddata = graph.getData(dst, flag);
             if(ddata == BFS::DIST_INFINITY) {
                 for (auto e : graph.in_edges(dst)) {
                   auto src = graph.getInEdgeDst(e);

                   if (front_bitset.test(src)) {
                     /*
                      * Currently assigning parents on the bfs path.
                      * Assign nextLevel (uncomment below) 
                      */
                     //ddata = nextLevel;
                     ddata = src;
                     next_bitset.set(dst);
                     work_items += 1;
                     break;
                   }
                 }
               }
             },
             galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
             galois::loopname((std::string("Sync-pull_") + std::to_string(runID)).c_str()));

            std::swap(front_bitset, next_bitset);
            next_bitset.reset();
      } while(work_items.reduce() >= old_workItemNum || (work_items.reduce() > numNodes / beta));

      BitsetToWl(graph, front_bitset, *next);
      scout_count = 1;
    }
    else {
      //c_push++;
      ++nextLevel;
      edges_to_check -= scout_count;
      work_items.reset();
      //PUSH to out-edges
      loop(galois::iterate(*curr),
           [&](const T& src) {
             for (auto e : graph.edges(src)) {
               auto dst = graph.getEdgeDst(e);
               auto& ddata = graph.getData(dst, flag);

               if (ddata == BFS::DIST_INFINITY) {
                 Dist oldDist = ddata;
                    /* 
                     * Currently assigning parents on the bfs path.
                     * Assign nextLevel (uncomment below) 
                     */
                   //if(__sync_bool_compare_and_swap(&ddata, oldDist, nextLevel)) {
                   if(__sync_bool_compare_and_swap(&ddata, oldDist, src)) {
                     next->push(dst);
                     work_items += (graph.edge_end(dst) - graph.edge_begin(dst));
                   }
               }
             }
           },
           galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
           galois::loopname((std::string("Sync-push_") + std::to_string(runID)).c_str()));

        scout_count = work_items.reduce();
    }
  }

  delete curr;
  delete next;
}

template <bool CONCURRENT, typename T, typename P, typename R>
void asyncAlgo(Graph& graph, GNode source, const P& pushWrap,
               const R& GALOIS_UNUSED(edgeRange)) {

  namespace gwl = galois::worklists;
  // typedef PerSocketChunkFIFO<CHUNK_SIZE> dFIFO;
  using FIFO = gwl::PerSocketChunkFIFO<CHUNK_SIZE>;
  using WL   = FIFO;

  using Loop =
      typename std::conditional<CONCURRENT, galois::ForEach,
                                galois::WhileQ<galois::SerFIFO<T>>>::type;

  Loop loop;

  galois::GAccumulator<size_t> BadWork;
  galois::GAccumulator<size_t> WLEmptyWork;

  graph.getData(source) = 0;
  galois::InsertBag<T> initBag;

  if (CONCURRENT) {
    pushWrap(initBag, source, "parallel");
  } else {
    pushWrap(initBag, source);
  }

  loop(galois::iterate(initBag),
       [&](const GNode& src, auto& ctx) {
         constexpr galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;

         for (auto ii : graph.edges(src)) {
           GNode dst   = graph.getEdgeDst(ii);
           auto& ddata = graph.getData(dst, flag);


	if (ddata == BFS::DIST_INFINITY) {
            Dist oldDist = ddata;
            if(__sync_bool_compare_and_swap(&ddata, oldDist, src)) {
                ctx.push(dst);
             }
           }
         }
       },
       galois::wl<WL>(), galois::loopname("runBFS"), galois::no_conflicts());
}


template <bool CONCURRENT>
void runAlgo(Graph& graph, const GNode& source, const uint32_t runID) {

  switch (algo) {
  case SyncDO:
    syncDOAlgo<CONCURRENT, GNode>(graph, source, NodePushWrap(),
                                   OutEdgeRangeFn{graph}, runID);
    break;
  case Async:
    asyncAlgo<CONCURRENT, GNode>(graph, source, NodePushWrap(),
                                         OutEdgeRangeFn{graph});
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

  galois::StatTimer StatTimer_graphConstuct("TimerConstructGraph", "BFS");
  StatTimer_graphConstuct.start();
  graph.readAndConstructBiGraphFromGRFile(filename);
  StatTimer_graphConstuct.stop();
  std::cout << "Read " << graph.size() << " nodes, " << graph.sizeEdges()
            << " edges" << std::endl;

  if (startNode >= graph.size() || reportNode >= graph.size()) {
    std::cerr << "failed to set report: " << reportNode
              << " or failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }

  auto it = graph.begin();
  std::advance(it, static_cast<unsigned long long>(startNode));
  source = *it;
  it     = graph.begin();
  std::advance(it, static_cast<unsigned long long>(reportNode));
  report = *it;

  galois::preAlloc(preAlloc);
  galois::gPrint("Fixed preAlloc done : ", preAlloc,"\n");
  galois::reportPageAlloc("MeminfoPre");

  galois::do_all(galois::iterate(graph),
                 [&graph](GNode n) { graph.getData(n) = BFS::DIST_INFINITY; });

  graph.getData(source) = 0;

  std::cout << "Running " << ALGO_NAMES[algo] << " algorithm with "
            << (bool(execution) ? "PARALLEL" : "SERIAL") << " execution "
            << std::endl;

  std::cout << "WARNING: This bfs version uses bi-directional CSR graph "
            << "and assigns parent instead of the shortest distance from source\n";          
  if(algo == Async) {
    std::cout << "WARNING: Async bfs does not use direction optimization. " 
              << "It uses Galois for_each for asynchronous execution which is advantageous " 
              << "for large diameter graphs such as road networks\n";
  }
  
  std::cout << " Execution started\n";
  galois::StatTimer Tmain;
  Tmain.start();

  for (unsigned int run = 0; run < numRuns; ++run) {
    galois::gPrint("BFS::go run ", run, " called\n");
    std::string timer_str("Timer_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), "BFS");
    StatTimer_main.start();

    if (execution == SERIAL) {
      runAlgo<false>(graph, source, run);
    } else if (execution == PARALLEL) {
      galois::runtime::profileVtune(
      [&]() {
      runAlgo<true>(graph, source, run);
      },"runAlgo");
    } else {
      std::cerr << "ERROR: unknown type of execution passed to -exec"
        << std::endl;
      std::abort();
    }

    StatTimer_main.stop();

    if ((run + 1) != numRuns) {
      for(unsigned int i = 0; i < 1; ++i) {
        galois::do_all(galois::iterate(graph),
            [&graph](GNode n) { graph.getData(n) = BFS::DIST_INFINITY; });
      }
    }
  }
  Tmain.stop();
  galois::reportPageAlloc("MeminfoPost");

  std::cout << "Node " << reportNode << " has parent "
            << graph.getData(report) << "\n";

  if (!skipVerify) {
    for(GNode n = 0; n < numPrint; n++){
    galois::gPrint("parent[", n, "] : ", graph.getData(n), "\n"); 
    }
  }

  return 0;
}
