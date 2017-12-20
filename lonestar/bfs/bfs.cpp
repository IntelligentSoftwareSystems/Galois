/** Breadth-first search -*- C++ -*-
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
 * Breadth-first search.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>
#include <deque>
#include <type_traits>

namespace cll = llvm::cl;

static const char* name = "Breadth-first Search";

static const char* desc =
  "Computes the shortest path from a source node to all nodes in a directed "
  "graph using a modified Bellman-Ford algorithm";

static const char* url = "breadth_first_search";

static cll::opt<std::string> filename(cll::Positional, 
                                      cll::desc("<input graph>"), 
                                      cll::Required);

static cll::opt<unsigned int> startNode("startNode",
                                        cll::desc("Node to start search from"),
                                        cll::init(0));
static cll::opt<unsigned int> reportNode("reportNode", 
                                         cll::desc("Node to report distance to"),
                                         cll::init(1));
static cll::opt<int> stepShift("delta",
                               cll::desc("Shift value for the deltastep"),
                               cll::init(10));
enum Algo {
  SyncTiled,
  Sync,
  Async,
  Serial
};

static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumVal(SyncTiled, "SyncTiled"),
      clEnumVal(Sync, "Sync"),
      clEnumVal(Async, "Async"),
      clEnumVal(Serial, "Serial"),
      clEnumValEnd), cll::init(SyncTiled));


using Graph = galois::graphs::LC_CSR_Graph<unsigned, void>
  ::with_no_lockable<true>::type
  ::with_numa_alloc<true>::type;

using GNode =  Graph::GraphNode;


#include "Lonestar/BFS_SSSP.h"


constexpr static const bool TRACK_WORK = false;
constexpr static const unsigned CHUNK_SIZE = 32u;



void syncTiledAlgo(Graph& graph, GNode source) {

  struct EdgeTile {
    Graph::edge_iterator beg;
    Graph::edge_iterator end;
  };

  constexpr galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
  constexpr ptrdiff_t EDGE_TILE_SIZE = 1024;


  Dist nextLevel = 0u;
  graph.getData(source, flag) = 0u;

  galois::InsertBag<GNode> activeNodes;
  galois::InsertBag<EdgeTile> edgeTiles;

  activeNodes.push(source);

  while (!activeNodes.empty()) {

    galois::do_all(galois::iterate(activeNodes),
        [&] (const GNode& src) {
          auto beg = graph.edge_begin(src, flag);
          const auto end = graph.edge_end(src, flag);

          assert(beg <= end);

          if ((end - beg) > EDGE_TILE_SIZE) {
            for (; beg + EDGE_TILE_SIZE < end;) {
              auto ne = beg + EDGE_TILE_SIZE;
              edgeTiles.push( EdgeTile{beg, ne} );
              beg = ne;
            }
          }
          
          if ((end - beg) > 0) {
            edgeTiles.push( EdgeTile{beg, end} );
          }
        
        },
        galois::steal(),
        galois::chunk_size<CHUNK_SIZE>(),
        galois::loopname("activeNodes"));

    ++nextLevel;
    activeNodes.clear_parallel();

    galois::do_all(galois::iterate(edgeTiles),
        [&] (const EdgeTile& tile) {

          for (auto e = tile.beg; e != tile.end; ++e) {
            auto dst = graph.getEdgeDst(e);
            auto& dstData = graph.getData(dst, flag);

            if (dstData == DIST_INFINITY) {
              dstData = nextLevel;
              activeNodes.push(dst);
            }
          }
        },
        galois::steal(),
        galois::chunk_size<CHUNK_SIZE>(),
        galois::loopname("edgeTiles"));

    edgeTiles.clear_parallel();
  }
}

void syncAlgo(Graph& graph, GNode source) {

  using Bag = galois::InsertBag<GNode>;
  constexpr galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;

  Bag* curr = new Bag();
  Bag* next = new Bag();

  Dist nextLevel = 0u;
  graph.getData(source, flag) = 0u;
  next->push(source);

  while (!next->empty()) {

    std::swap(curr, next);
    next->clear_parallel();
    ++nextLevel;

    galois::do_all(galois::iterate(*curr),
        [&] (const GNode& src) {

          for (auto e: graph.edges(src, flag)) {
            auto dst = graph.getEdgeDst(e);
            auto& dstData = graph.getData(dst, flag);

            if (dstData == DIST_INFINITY) {
              dstData = nextLevel;
              next->push(dst);
            }
          }
        },
        galois::steal(),
        galois::chunk_size<CHUNK_SIZE>(),
        galois::loopname("Sync"));
  }


  delete curr;
  delete next;
}


void asyncAlgo(Graph& graph, GNode source) {

  using namespace galois::worklists;
  typedef dChunkedFIFO<CHUNK_SIZE> dFIFO;
  typedef BulkSynchronous<dChunkedLIFO<CHUNK_SIZE> > BSWL;
  using WL = BSWL;

  constexpr bool useCAS = !std::is_same<WL, BSWL>::value;

  galois::GAccumulator<size_t> BadWork;
  galois::GAccumulator<size_t> WLEmptyWork;

  graph.getData(source) = 0;

  galois::for_each(galois::iterate({ UpdateRequest{source, 1} })
      , [&] (const UpdateRequest& req, auto& ctx) {
        constexpr galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
        Dist sdist = graph.getData(req.n, flag);
        
        if (TRACK_WORK) {
          if (req.w != sdist) {
            WLEmptyWork += 1;
            return;
          }
        }

        Dist newDist = req.w;

        for (auto ii : graph.edges(req.n, flag)) {
          GNode dst = graph.getEdgeDst(ii);
          auto& ddata  = graph.getData(dst, flag);

          while (true) {

            Dist oldDist = ddata;

            if (oldDist <= newDist) {
              break;
            }

            if (!useCAS || __sync_bool_compare_and_swap(&ddata, oldDist, newDist)) {

              if (!useCAS) {
                ddata = newDist;
              }

              if (TRACK_WORK) {
                if (oldDist != DIST_INFINITY) {
                   BadWork += 1;
                }
              }

              ctx.push( UpdateRequest(dst, newDist + 1) );
              break;
            }
          }
        } // end for
      }
      , galois::wl<BSWL>()
      , galois::loopname("runBFS")
      , galois::no_conflicts());

  if (TRACK_WORK) {
    galois::runtime::reportStat_Single("BFS", "BadWork", BadWork.reduce());
    galois::runtime::reportStat_Single("BFS", "EmptyWork", WLEmptyWork.reduce());
  }
}

void serialAlgo(Graph& graph, GNode source) {

  using WL = std::deque<UpdateRequest>;
  constexpr galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;

  WL wl;

  graph.getData(source, flag) = 0;
  wl.push_back(UpdateRequest(source, 1));

  size_t iter = 0;

  while (!wl.empty()) {
    ++iter;

    UpdateRequest req = wl.front();
    wl.pop_front();

    for (auto e: graph.edges(req.n, flag)) {

      auto dst = graph.getEdgeDst(e);
      auto& dstData = graph.getData(dst, flag);

      if (dstData == DIST_INFINITY) {
        dstData = req.w;
        wl.push_back(UpdateRequest(dst, req.w + 1));
      }
    }

  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer T("OverheadTime");
  T.start();
  
  Graph graph;
  GNode source, report;

  std::cout << "Reading from file: " << filename << std::endl;
  galois::graphs::readGraph(graph, filename); 
  std::cout << "Read " << graph.size() << " nodes, "
    << graph.sizeEdges() << " edges" << std::endl;

  if (startNode >= graph.size() || reportNode >= graph.size()) {
    std::cerr << "failed to set report: " << reportNode
              << " or failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }

  auto it = graph.begin();
  std::advance(it, startNode);
  source = *it;
  it = graph.begin();
  std::advance(it, reportNode);
  report = *it;

  size_t approxNodeData = graph.size() * 64;
  // size_t approxEdgeData = graph.sizeEdges() * sizeof(typename
  // Graph::edge_data_type) * 2;
  galois::preAlloc(8*numThreads +
                   approxNodeData / galois::runtime::pagePoolSize());

  galois::reportPageAlloc("MeminfoPre");

  galois::do_all(galois::iterate(graph), 
                       [&graph] (GNode n) { graph.getData(n) = DIST_INFINITY; });
  graph.getData(source) = 0;

  galois::StatTimer Tmain;
  Tmain.start();

  switch(algo) {
    case SyncTiled:
      std::cout << "Running SyncTiled algorithm\n";
      syncTiledAlgo(graph, source);
      break;
    case Sync:
      std::cout << "Running Sync algorithm\n";
      syncAlgo(graph, source);
      break;
    case Async:
      std::cout << "Running Async algorithm\n";
      asyncAlgo(graph, source);
      break;
    case Serial:
      std::cout << "Running Serial algorithm\n";
      serialAlgo(graph, source);
      break;
    default:
      std::abort();
  }

  Tmain.stop();
  T.stop();

  galois::reportPageAlloc("MeminfoPost");
  galois::runtime::reportNumaAlloc("NumaPost");
  
  std::cout << "Node " << reportNode << " has distance "
            << graph.getData(report) << "\n";

  if (!skipVerify) {
    if (verify<true>(graph, source)) {
      std::cout << "Verification successful.\n";
    } else {
      GALOIS_DIE("Verification failed");
    }
  }


  return 0;
}
