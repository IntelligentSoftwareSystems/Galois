/** Page rank application -*- C++ -*-
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
 * @author Joyce Whang <joyce@cs.utexas.edu>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#include "Lonestar/BoilerPlate.h"
#include "PageRank-constants.h"
#include "galois/Bag.h"
#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"

// These implementations are based on the Push-based PageRank computation
// (Algorithm 4) as described in the PageRank Europar 2015 paper.

namespace cll = llvm::cl;
const char* desc =
    "Computes page ranks a la Page and Brin. This is a push-style algorithm.";

constexpr static const unsigned CHUNK_SIZE = 16;

enum Algo { Async, Sync };

static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
                           cll::values(clEnumVal(Async, "Async"),
                                       clEnumVal(Sync, "Sync"), clEnumValEnd),
                           cll::init(Sync));

static cll::opt<std::string>
    filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"),
                                 cll::init(TOLERANCE));
static cll::opt<unsigned int>
    maxIterations("maxIterations",
                  cll::desc("Maximum iterations, Sync version only"),
                  cll::init(MAX_ITER));

struct LNode {
  PRTy value;
  std::atomic<PRTy> residual;

  void init() {
    value    = 0.0;
    residual = 1 - ALPHA;
  }

  friend std::ostream& operator<<(std::ostream& os, const LNode& n) {
    os << "{PR " << n.value << ", residual " << n.residual << "}";
    return os;
  }
};

typedef galois::graphs::LC_CSR_Graph<LNode, void>::with_numa_alloc<
    true>::type ::with_no_lockable<true>::type Graph;
typedef typename Graph::GraphNode GNode;

template <typename Graph>
static void printTop(Graph& graph, int topn) {
  typedef typename Graph::node_data_reference node_data_reference;
  typedef TopPair<GNode> Pair;
  typedef std::map<Pair, GNode> Top;

  Top top;

  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src             = *ii;
    node_data_reference n = graph.getData(src);
    float value           = n.value;
    Pair key(value, src);

    if ((int)top.size() < topn) {
      top.insert(std::make_pair(key, src));
      continue;
    }

    if (top.begin()->first < key) {
      top.erase(top.begin());
      top.insert(std::make_pair(key, src));
    }
  }

  int rank = 1;
  std::cout << "Rank PageRank Id\n";
  for (typename Top::reverse_iterator ii = top.rbegin(), ei = top.rend();
       ii != ei; ++ii, ++rank) {
    std::cout << rank << ": " << ii->first.value << " " << ii->first.id << "\n";
  }
}

// void initResidual(Graph& graph) {
//   galois::do_all(
//       galois::iterate(graph),
//       [&graph](const GNode& src) {
//         auto nout = std::distance(graph.edge_begin(src),
//         graph.edge_end(src)); for (auto ii : graph.edges(src)) {
//           auto dst    = graph.getEdgeDst(ii);
//           auto& ddata = graph.getData(dst);
//           atomicAdd(ddata.residual, 1 / nout);
//         }
//       },
//       galois::loopname("initResidual"), galois::steal(), galois::no_stats());

//   galois::do_all(galois::iterate(graph),
//                  [&graph](const GNode& src) {
//                    auto& data    = graph.getData(src);
//                    data.residual = data.residual * ALPHA * (1.0 - ALPHA);
//                  },
//                  galois::loopname("scaleResidual"), galois::steal(),
//                  galois::no_stats());
// }

void asyncPageRank(Graph& graph) {
  typedef galois::worklists::dChunkedFIFO<CHUNK_SIZE> WL;
  galois::for_each(galois::iterate(graph),
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
                         // for each out-going neighbors
                         for (auto jj : graph.edges(src, flag)) {
                           GNode dst    = graph.getEdgeDst(jj);
                           LNode& ddata = graph.getData(dst, flag);
                           if (delta > 0) {
                             auto old = atomicAdd(ddata.residual, delta);
                             if ((old < tolerance) &&
                                 (old + delta >= tolerance)) {
                               ctx.push(dst);
                             }
                           }
                         }
                       }
                     }
                   },
                   galois::loopname("AsyncPageRank"), galois::no_conflicts(),
                   galois::no_stats(), galois::wl<WL>());
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

  galois::do_all(galois::iterate(graph),
                 [&](const GNode& src) { activeNodes.push(src); },
                 galois::no_stats());

  size_t iter = 0;
  for (; !activeNodes.empty() && iter < maxIterations; ++iter) {

    galois::do_all(galois::iterate(activeNodes),
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

                       // Edge tiling for large outdegree nodes
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

    activeNodes.clear_parallel();

    galois::do_all(galois::iterate(updates),
                   [&](const Update& up) {
                     constexpr const galois::MethodFlag flag =
                         galois::MethodFlag::UNPROTECTED;
                     // for each out-going neighbors
                     for (auto jj = up.beg; jj != up.end; ++jj) {
                       GNode dst    = graph.getEdgeDst(jj);
                       LNode& ddata = graph.getData(dst, flag);
                       auto old     = atomicAdd(ddata.residual, up.delta);
                       // if fabs(old) is greater than tolerance, then it would
                       // already have been processed in the previous do_all
                       // loop
                       if ((old <= tolerance) &&
                           (old + up.delta >= tolerance)) {
                         activeNodes.push(dst);
                       }
                     }
                   },
                   galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
                   galois::loopname("PushResidual"), galois::no_stats());

    updates.clear_parallel();
  }

  if (iter >= maxIterations) {
    std::cerr << "ERROR: failed to converge in " << iter << " iterations"
              << std::endl;
  }
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer T("OverheadTime");
  T.start();

  Graph graph;
  galois::graphs::readGraph(graph, filename);
  std::cout << "Read " << graph.size() << " nodes, " << graph.sizeEdges()
            << " edges\n";

  galois::preAlloc(numThreads +
                   (5 * graph.size() * sizeof(typename Graph::node_data_type)) /
                       galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  switch (algo) {
  case Async:
    std::cout << "Running Edge Async push version,";
    break;
  case Sync:
    std::cout << "Running Edge Sync push version,";
    break;
  default:
    std::abort();
  }
  std::cout << "tolerance:" << tolerance << ", maxIterations:" << maxIterations
            << "\n";

  galois::do_all(galois::iterate(graph),
                 [&graph](GNode n) { graph.getData(n).init(); },
                 galois::no_stats(), galois::loopname("Initialize"));
  // initResidual(graph);

  galois::StatTimer Tmain;
  Tmain.start();

  switch (algo) {
  case Async: {
    asyncPageRank(graph);
    break;
  }
  case Sync: {
    syncPageRank(graph);
  }
  }

  Tmain.stop();

  galois::reportPageAlloc("MeminfoPost");

  if (!skipVerify) {
    printTop(graph, PRINT_TOP);
  }

  T.stop();

  return 0;
}
