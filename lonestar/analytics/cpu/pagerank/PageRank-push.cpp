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

#include "Lonestar/BoilerPlate.h"
#include "PageRank-constants.h"
#include "galois/Bag.h"
#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"

/**
 * These implementations are based on the Push-based PageRank computation
 * (Algorithm 4) as described in the PageRank Europar 2015 paper.
 *
 * WHANG, Joyce Jiyoung, et al. Scalable data-driven pagerank: Algorithms,
 * system issues, and lessons learned. In: European Conference on Parallel
 * Processing. Springer, Berlin, Heidelberg, 2015. p. 438-450.
 */

const char* desc =
    "Computes page ranks a la Page and Brin. This is a push-style algorithm.";

constexpr static const unsigned CHUNK_SIZE = 16;

enum Algo { Async, Sync }; ///< Async has better asbolute performance.

static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
                           cll::values(clEnumVal(Async, "Async"),
                                       clEnumVal(Sync, "Sync")),
                           cll::init(Async));

struct LNode {
  PRTy value;
  std::atomic<PRTy> residual;

  void init() {
    value    = 0.0;
    residual = INIT_RESIDUAL;
  }

  friend std::ostream& operator<<(std::ostream& os, const LNode& n) {
    os << "{PR " << n.value << ", residual " << n.residual << "}";
    return os;
  }
};

typedef galois::graphs::LC_CSR_Graph<LNode, void>::with_numa_alloc<
    true>::type ::with_no_lockable<true>::type Graph;
typedef typename Graph::GraphNode GNode;

void asyncPageRank(Graph& graph) {
  typedef galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE> WL;
  galois::for_each(
      galois::iterate(graph),
      [&](GNode src, auto& ctx) {
        LNode& sdata = graph.getData(src);
        constexpr const galois::MethodFlag flag =
            galois::MethodFlag::UNPROTECTED;

        if (sdata.residual > tolerance) {
          PRTy oldResidual = sdata.residual.exchange(0.0);
          sdata.value += oldResidual;
          int src_nout = std::distance(graph.edge_begin(src, flag),
                                       graph.edge_end(src, flag));
          if (src_nout > 0) {
            PRTy delta = oldResidual * ALPHA / src_nout;
            //! For each out-going neighbors.
            for (auto jj : graph.edges(src, flag)) {
              GNode dst    = graph.getEdgeDst(jj);
              LNode& ddata = graph.getData(dst, flag);
              if (delta > 0) {
                auto old = atomicAdd(ddata.residual, delta);
                if ((old < tolerance) && (old + delta >= tolerance)) {
                  ctx.push(dst);
                }
              }
            }
          }
        }
      },
      galois::loopname("PushResidualAsync"),
      galois::disable_conflict_detection(), galois::no_stats(),
      galois::wl<WL>());
}

void syncPageRank(Graph& graph) {
  struct Update {
    PRTy delta;
    Graph::edge_iterator beg;
    Graph::edge_iterator end;
  };

  constexpr ptrdiff_t EDGE_TILE_SIZE = 128;

  galois::InsertBag<Update> updates;
  galois::InsertBag<GNode> activeNodes;

  galois::do_all(
      galois::iterate(graph), [&](const GNode& src) { activeNodes.push(src); },
      galois::no_stats());

  size_t iter = 0;
  for (; !activeNodes.empty() && iter < maxIterations; ++iter) {
    galois::do_all(
        galois::iterate(activeNodes),
        [&](const GNode& src) {
          constexpr const galois::MethodFlag flag =
              galois::MethodFlag::UNPROTECTED;
          LNode& sdata = graph.getData(src, flag);

          if (sdata.residual > tolerance) {
            PRTy oldResidual = sdata.residual;
            sdata.value += oldResidual;
            sdata.residual = 0.0;

            int src_nout = std::distance(graph.edge_begin(src, flag),
                                         graph.edge_end(src, flag));
            PRTy delta   = oldResidual * ALPHA / src_nout;

            auto beg       = graph.edge_begin(src, flag);
            const auto end = graph.edge_end(src, flag);

            assert(beg <= end);

            //! Edge tiling for large outdegree nodes.
            if ((end - beg) > EDGE_TILE_SIZE) {
              for (; beg + EDGE_TILE_SIZE < end;) {
                auto ne = beg + EDGE_TILE_SIZE;
                updates.push(Update{delta, beg, ne});
                beg = ne;
              }
            }

            if ((end - beg) > 0) {
              updates.push(Update{delta, beg, end});
            }
          }
        },
        galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
        galois::loopname("CreateEdgeTiles"), galois::no_stats());

    activeNodes.clear();

    galois::do_all(
        galois::iterate(updates),
        [&](const Update& up) {
          constexpr const galois::MethodFlag flag =
              galois::MethodFlag::UNPROTECTED;
          //! For each out-going neighbors.
          for (auto jj = up.beg; jj != up.end; ++jj) {
            GNode dst    = graph.getEdgeDst(jj);
            LNode& ddata = graph.getData(dst, flag);
            auto old     = atomicAdd(ddata.residual, up.delta);
            //! If fabs(old) is greater than tolerance, then it would
            //! already have been processed in the previous do_all
            //! loop.
            if ((old <= tolerance) && (old + up.delta >= tolerance)) {
              activeNodes.push(dst);
            }
          }
        },
        galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
        galois::loopname("PushResidualSync"), galois::no_stats());

    updates.clear();
  }

  if (iter >= maxIterations) {
    std::cerr << "ERROR: failed to converge in " << iter << " iterations\n";
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url, &inputFile);

  galois::StatTimer totalTime("TimerTotal");
  totalTime.start();

  Graph graph;
  galois::graphs::readGraph(graph, inputFile);
  std::cout << "Read " << graph.size() << " nodes, " << graph.sizeEdges()
            << " edges\n";

  galois::preAlloc(5 * numThreads +
                   (5 * graph.size() * sizeof(typename Graph::node_data_type)) /
                       galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  std::cout << "tolerance:" << tolerance << ", maxIterations:" << maxIterations
            << "\n";

  galois::do_all(
      galois::iterate(graph), [&graph](GNode n) { graph.getData(n).init(); },
      galois::no_stats(), galois::loopname("Initialize"));

  galois::StatTimer execTime("Timer_0");
  execTime.start();

  switch (algo) {
  case Async:
    std::cout << "Running Edge Async push version,";
    asyncPageRank(graph);
    break;

  case Sync:
    std::cout << "Running Edge Sync push version,";
    syncPageRank(graph);
    break;

  default:
    std::abort();
  }

  execTime.stop();

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    printTop(graph);
  }

#if DEBUG
  printPageRank(graph);
#endif

  totalTime.stop();

  return 0;
}
