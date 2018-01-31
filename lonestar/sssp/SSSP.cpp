/** Single source shortest paths -*- C++ -*-
 * @example SSSP.cpp
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
 * Single source shortest paths.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/PriorityQueue.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"
#include "Lonestar/BFS_SSSP.h"

#include <iostream>

namespace cll = llvm::cl;

static const char* name = "Single Source Shortest Path";
static const char* desc =
    "Computes the shortest path from a source node to all nodes in a directed "
    "graph using a modified chaotic iteration algorithm";
static const char* url = "single_source_shortest_path";

static cll::opt<std::string> filename(cll::Positional, 
                                      cll::desc("<input graph>"), 
                                      cll::Required);

static cll::opt<unsigned int> startNode("startNode",
                                        cll::desc("Node to start search from"),
                                        cll::init(0));
static cll::opt<unsigned int> reportNode("reportNode", 
                                         cll::desc("Node to report distance to"),
                                         cll::init(1));
static cll::opt<unsigned int> stepShift("delta",
                               cll::desc("Shift value for the deltastep"),
                               cll::init(10));

enum Algo {
  deltaStep,
  deltaTiled,
  serDelta,
  serDeltaTiled,
  dijkstra
};

static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumVal(deltaStep, "deltaStep"),
      clEnumVal(deltaTiled, "deltaTiled"),
      clEnumVal(serDelta, "serDelta"),
      clEnumVal(dijkstra, "dijkstra"),
      clEnumVal(serDeltaTiled, "serDeltaTiled"),
      clEnumValEnd), cll::init(deltaTiled));

// typedef galois::graphs::LC_InlineEdge_Graph<std::atomic<unsigned int>, uint32_t>::with_no_lockable<true>::type::with_numa_alloc<true>::type Graph;
using Graph = galois::graphs::LC_CSR_Graph<std::atomic<uint32_t>, uint32_t>
  ::with_no_lockable<true>::type
  ::with_numa_alloc<true>::type;
typedef Graph::GraphNode GNode;

constexpr static const bool TRACK_WORK = false;
constexpr static const unsigned CHUNK_SIZE = 64u;
constexpr static const ptrdiff_t EDGE_TILE_SIZE = 4096;

using SSSP = BFS_SSSP<Graph, uint32_t, true, EDGE_TILE_SIZE>;
using Dist = SSSP::Dist;
using UpdateRequest = SSSP::UpdateRequest;
using UpdateRequestIndexer = SSSP::UpdateRequestIndexer;
using DistEdgeTile = SSSP::DistEdgeTile;
using DistEdgeTileIndexer = SSSP::DistEdgeTileIndexer;
using DistEdgeTileMaker = SSSP::DistEdgeTileMaker;

void deltaStepAlgo(Graph& graph, const GNode& source) {

  galois::GAccumulator<size_t> BadWork;
  galois::GAccumulator<size_t> WLEmptyWork;

  namespace gwl = galois::worklists;

  using dChunk = gwl:: dChunkedFIFO<CHUNK_SIZE>;
  using OBIM = gwl::OrderedByIntegerMetric<UpdateRequestIndexer, dChunk>;

  graph.getData(source) = 0;

  galois::for_each(galois::iterate( { UpdateRequest{source, 0} } ),
      [&] (const UpdateRequest& req, auto& ctx) {
        const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;
        const auto& sdata = graph.getData(req.n, flag);
        
        if (req.w != sdata) {
          if (TRACK_WORK)
            WLEmptyWork += 1;
          return;
        }
        
        for (auto ii : graph.edges(req.n, flag)) {
          GNode dst = graph.getEdgeDst(ii);
          auto& ddist  = graph.getData(dst, flag);
          Dist ew    = graph.getEdgeData(ii, flag);
          while (true) {
            Dist oldDist = ddist;
            Dist newDist = sdata + ew;

            if (oldDist <= newDist) { break; }

            if (ddist.compare_exchange_weak(oldDist, newDist, std::memory_order_relaxed)) {
              ctx.push( UpdateRequest(dst, newDist) );
              break;
            }
          }
        }
      },
      // galois::wl<OBIM>( UpdateRequestIndexer{stepShift} ), 
      galois::wl<gwl::ParaMeter<> >(),
      galois::no_conflicts(), 
      galois::loopname("SSSP"));

  if (TRACK_WORK) {
    galois::runtime::reportStat_Single("SSSP", "BadWork", BadWork.reduce());
    galois::runtime::reportStat_Single("SSSP", "WLEmptyWork", WLEmptyWork.reduce()); 
  }

}

void serialDeltaAlgo(Graph& graph, const GNode& source) {

  SerialBucketWL<UpdateRequest, UpdateRequestIndexer> wl(UpdateRequestIndexer {stepShift});;
  graph.getData(source) = 0;

  wl.push_back( UpdateRequest{source, 0} );

  size_t iter = 0ul;
  while (!wl.empty()) {

    auto& curr = wl.minBucket();

    while (!curr.empty()) {
      ++iter;
      UpdateRequest req = curr.front();
      curr.pop_front();

      if (graph.getData(req.n) < req.w) {
        // empty work
        continue;
      }

      for (auto e: graph.edges(req.n)) {

        GNode dst = graph.getEdgeDst(e);
        auto& ddata = graph.getData(dst);

        auto newDist = req.w + graph.getEdgeData(e);

        if (newDist < ddata) {
          ddata = newDist;
          wl.push_back( UpdateRequest(dst, newDist) );
        }
      }
    }

    wl.goToNextBucket();
  }

  if (!wl.allEmpty()) { std::abort(); }
  galois::runtime::reportStat_Single("SSSP-Serial-Delta", "Iterations", iter);
}

struct SrcEdgeTile {
  GNode src;
  Dist dist;
  Graph::edge_iterator beg;
  Graph::edge_iterator end;
};

struct SrcEdgeTileMaker {
  GNode src;
  Dist dist;

  SrcEdgeTile operator () (const Graph::edge_iterator& beg, const Graph::edge_iterator& end) const {
    return SrcEdgeTile {src, dist, beg, end};
  }
};

void deltaStepTiledAlgo(Graph& graph, const GNode& source) {

  namespace gwl = galois::worklists;

  using dChunk = gwl:: dChunkedFIFO<CHUNK_SIZE>;
  using OBIM = gwl::OrderedByIntegerMetric<DistEdgeTileIndexer, dChunk>;

  graph.getData(source) = 0;

  galois::InsertBag<SrcEdgeTile> initBag;

  SSSP::pushEdgeTiles(initBag, graph, source, SrcEdgeTileMaker {source, 0u} );

  galois::for_each(galois::iterate(initBag),
      [&] (const SrcEdgeTile& tile, auto& ctx) {

        const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;

        const auto& sdata = graph.getData(tile.src, flag);
        
        if (sdata < tile.dist) {
          // empty work;
          return;
        }

        for (auto ii = tile.beg; ii != tile.end; ++ii) {

          GNode dst = graph.getEdgeDst(ii);
          auto& ddist  = graph.getData(dst, flag);
          Dist ew = graph.getEdgeData(ii, flag);

          while (true) {
            Dist oldDist = ddist;
            Dist newDist = sdata + ew;

            if (oldDist <= newDist) { break; }

            if (ddist.compare_exchange_weak(oldDist, newDist, std::memory_order_relaxed)) {
              SSSP::pushEdgeTiles(ctx, graph, dst, SrcEdgeTileMaker{dst, newDist} );
              break;
            }
          }
        }
      },
      galois::wl<OBIM>( DistEdgeTileIndexer{stepShift} ), 
      galois::no_conflicts(), 
      galois::loopname("SSSP"));

}

void serialDeltaTiledAlgo(Graph& graph, const GNode& source) {

  SerialBucketWL<SrcEdgeTile, SSSP::DistEdgeTileIndexer> wl(DistEdgeTileIndexer {stepShift});;
  graph.getData(source) = 0;

  SSSP::pushEdgeTiles(wl, graph, source, SrcEdgeTileMaker{source, 0u} );

  size_t iter = 0ul;
  while (!wl.empty()) {

    auto& curr = wl.minBucket();

    while (!curr.empty()) {
      ++iter;
      SrcEdgeTile tile = curr.front();
      curr.pop_front();

      const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;

      auto& sdata = graph.getData(tile.src, flag);
      if (sdata < tile.dist) {
        // empty work;
        continue;
      }

      for (auto e = tile.beg; e != tile.end; ++e) {

        GNode dst = graph.getEdgeDst(e);
        auto& ddata = graph.getData(dst);

        auto newDist = sdata + graph.getEdgeData(e);

        if (newDist < ddata) {
          ddata = newDist;
          SSSP::pushEdgeTiles(wl, graph, dst, SrcEdgeTileMaker{dst, newDist} );
        }
      }
    }

    wl.goToNextBucket();
  }

  if (!wl.allEmpty()) { std::abort(); }
  galois::runtime::reportStat_Single("SSSP-Serial-Delta-Tiled", "Iterations", iter);
}

void dijkstraAlgo(Graph& graph, const GNode& source) {
  using WL = galois::MinHeap<UpdateRequest>;

  graph.getData(source) = 0;

  WL wl;
  wl.push( UpdateRequest(source, 0) );

  size_t iter = 0;

  while (!wl.empty()) {
    ++iter;

    UpdateRequest req = wl.pop();

    if (graph.getData(req.n) < req.w) {
      // empty work
      continue;
    }

    for (auto e: graph.edges(req.n)) {
      GNode dst = graph.getEdgeDst(e);
      auto& ddata = graph.getData(dst);

      auto newDist = req.w + graph.getEdgeData(e);

      if (newDist < ddata) {
        ddata = newDist;
        wl.push( UpdateRequest(dst, newDist) );
      }
    }

  }

  galois::runtime::reportStat_Single("SSSP-Dijkstra", "Iterations", iter);
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
  galois::preAlloc(numThreads +
                   approxNodeData / galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  std::cout << "INFO: Using delta-step of " << (1 << stepShift) << "\n";
  std::cout << "WARNING: Performance varies considerably due to delta parameter.\n";
  std::cout << "WARNING: Do not expect the default to be good for your graph.\n";
  galois::do_all(galois::iterate(graph), 
      [&graph] (GNode n) { 
        graph.getData(n) = SSSP::DIST_INFINITY; 
      });

  graph.getData(source) = 0;

  galois::StatTimer Tmain;
  Tmain.start();

  switch(algo) {
    case deltaTiled:
      std::cout << "Running deltaTiled algorithm\n";
      deltaStepTiledAlgo(graph, source);
      break;
    case deltaStep:
      std::cout << "Running deltaStep algorithm\n";
      deltaStepAlgo(graph, source);
      break;
    case serDelta:
      std::cout << "Running serDelta algorithm\n";
      serialDeltaAlgo(graph, source);
      break;
    case serDeltaTiled:
      std::cout << "Running serDeltaTiled algorithm\n";
      serialDeltaTiledAlgo(graph, source);
      break;
    case dijkstra:
      std::cout << "Running dijkstra algorithm\n";
      dijkstraAlgo(graph, source);
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
    if (SSSP::verify(graph, source)) {
      std::cout << "Verification successful.\n";
    } else {
      GALOIS_DIE("Verification failed");
    }
  }

  return 0;
}
