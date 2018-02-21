/** Page rank application -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
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
 */

#include "Lonestar/BoilerPlate.h"
#include "PageRank-constants.h"
#include "galois/Galois.h"
#include "galois/LargeArray.h"
#include "galois/PerThreadContainer.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "galois/runtime/Profile.h"

namespace cll = llvm::cl;
const char* desc =
    "Computes page ranks a la Page and Brin. This is a pull-style algorithm.";

constexpr static const unsigned CHUNK_SIZE = 32;

enum Algo { PR_Pull, PR_Pull_ET, PR_Pull_Profile };

static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm:"),
    cll::values(clEnumVal(PR_Pull, "PUll"),
                clEnumVal(PR_Pull_ET, "Pull with edge tiling"), clEnumValEnd),
    cll::init(PR_Pull));

// We require a transpose graph since this is a pull-style algorithm
static cll::opt<std::string> filename(cll::Positional,
                                      cll::desc("<tranpose of input graph>"),
                                      cll::Required);
// Any delta in pagerank computation across iterations that is greater than the
// tolerance means the computation has not yet converged.
static cll::opt<float> tolerance("tolerance", cll::desc("tolerance"),
                                 cll::init(TOLERANCE));
static cll::opt<unsigned int> maxIterations("maxIterations",
                                            cll::desc("Maximum iterations"),
                                            cll::init(MAX_ITER));

struct LNode {
  PRTy value[2]; // Final pagerank value is in value[1]
  uint32_t nout; // Compute the out degrees in the original graph

  PRTy getPageRank() const { return value[1]; }
  PRTy getPageRank(unsigned int it) const { return value[it & 1]; }
  void setPageRank(unsigned it, PRTy v) { value[(it + 1) & 1] = v; }
  void finalize(void) { value[1] = value[0]; }
};

typedef galois::graphs::LC_CSR_Graph<LNode, void>::with_no_lockable<
    true>::type ::with_numa_alloc<true>::type Graph;
typedef typename Graph::GraphNode GNode;

void initNodeData(Graph& g) {
  galois::do_all(galois::iterate(g),
                 [&](const GNode& n) {
                   auto& data = g.getData(n, galois::MethodFlag::UNPROTECTED);
                   data.value[0] = (1 - ALPHA);
                   data.value[1] = (1 - ALPHA);
                   data.nout     = 0;
                 },
                 galois::no_stats(), galois::loopname("initNodeData"));
}

// Computing outdegrees in the tranpose graph is equivalent to computing the
// indegrees in the original graph
void computeOutDeg(Graph& graph) {
  galois::StatTimer outDegreeTimer("computeOutDeg");
  outDegreeTimer.start();

  galois::LargeArray<std::atomic<size_t>> vec;
  vec.allocateInterleaved(graph.size());

  galois::do_all(galois::iterate(graph),
                 [&](const GNode& src) { vec.constructAt(src, 0ul); },
                 galois::no_stats(), galois::loopname("InitDegVec"));

  galois::do_all(galois::iterate(graph),
                 [&](const GNode& src) {
                   for (auto nbr : graph.edges(src)) {
                     GNode dst = graph.getEdgeDst(nbr);
                     vec[dst].fetch_add(1ul);
                   }
                 },
                 galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
                 galois::no_stats(), galois::loopname("ComputeDeg"));

  galois::do_all(galois::iterate(graph),
                 [&](const GNode& src) {
                   auto& srcData = graph.getData(src);
                   srcData.nout  = vec[src];
                 },
                 galois::no_stats(), galois::loopname("CopyDeg"));

  outDegreeTimer.stop();
}

/*
void computeOutDeg(Graph& graph) {
  galois::StatTimer t("computeOutDeg");
  t.start();

  galois::PerThreadVector<uint32_t> perThrdVecs;

  galois::on_each([&] (const unsigned tid, const unsigned numT) {
      perThrdVecs.get(tid).resize(graph.size(), 0u);
      });

  galois::do_all(galois::iterate(graph),
       [&] (const GNode& src) {
         for (auto nbr : graph.edges(src)) {
           GNode dst = graph.getEdgeDst(nbr);
           perThrdVecs.get()[dst] += 1;
         }
       },
       galois::steal(),
       galois::chunk_size<CHUNK_SIZE>(),
       galois::no_stats(),
       galois::loopname("ComputeDeg"));

  const unsigned numT = galois::getActiveThreads();

  galois::do_all(galois::iterate(graph),
      [&] (const GNode& src) {
        auto& srcData = graph.getData(src);
        for (unsigned i = 0; i < numT; ++i) {
          srcData.nout += perThrdVecs.get(i)[src];
        }
      },
      galois::no_stats(),
      galois::loopname("ReduceDeg"));

  t.stop();
}
*/

void finalizePR(Graph& g) {
  galois::do_all(galois::iterate(g),
                 [&](const GNode& n) {
                   LNode& data = g.getData(n, galois::MethodFlag::UNPROTECTED);
                   data.finalize();
                 },
                 galois::no_stats(), galois::loopname("Finalize"));
}

void computePageRankET(Graph& graph) {
  struct ActivePRNode {
    GNode src;
    bool needAtomic;
    Graph::edge_iterator beg;
    Graph::edge_iterator end;
  };

  constexpr ptrdiff_t EDGE_TILE_SIZE = 1024;

  galois::InsertBag<ActivePRNode> activeNodes;
  galois::LargeArray<std::atomic<PRTy>> atomicVec;
  atomicVec.allocateInterleaved(graph.size());
  galois::LargeArray<PRTy> nonAtomicVec;
  nonAtomicVec.allocateInterleaved(graph.size());

  galois::do_all(galois::iterate(graph),
                 [&](const GNode& src) {
                   atomicVec[src]    = 0;
                   nonAtomicVec[src] = 0;
                 },
                 galois::no_stats(), galois::loopname("InitSumsInVectors"));

  unsigned int iteration = 0;
  galois::GReduceMax<float> max_delta;

  // Split work not by nodes but by edges, to take into account high degree
  // nodes.
  galois::do_all(galois::iterate(graph),
                 [&](const GNode& src) {
                   constexpr const galois::MethodFlag flag =
                       galois::MethodFlag::UNPROTECTED;

                   auto beg       = graph.edge_begin(src, flag);
                   const auto end = graph.edge_end(src, flag);
                   assert(beg <= end);

                   bool needsAtomicStore = false;
                   // Edge tiling for large outdegree nodes
                   if ((end - beg) > EDGE_TILE_SIZE) {
                     needsAtomicStore = true;
                     for (; beg + EDGE_TILE_SIZE < end;) {
                       auto ne = beg + EDGE_TILE_SIZE;
                       activeNodes.push(ActivePRNode{src, true, beg, ne});
                       beg = ne;
                     }
                   }

                   if ((end - beg) > 0) {
                     activeNodes.push(ActivePRNode{
                         src, (needsAtomicStore) ? true : false, beg, end});
                   }

                 },
                 galois::no_stats(), galois::steal(),
                 galois::chunk_size<CHUNK_SIZE>(),
                 galois::loopname("SplitWorkByEdges"));

  while (true) {

    // Compute partial contributions to the final pagerank from the edge tiles
    galois::do_all(galois::iterate(activeNodes),
                   [&](const ActivePRNode& prNode) {
                     constexpr const galois::MethodFlag flag =
                         galois::MethodFlag::UNPROTECTED;
                     GNode src        = prNode.src;
                     PRTy partial_sum = 0.0;
                     for (auto ii = prNode.beg; ii != prNode.end; ++ii) {
                       GNode dst    = graph.getEdgeDst(ii);
                       LNode& ddata = graph.getData(dst, flag);
                       partial_sum += ddata.getPageRank(iteration) / ddata.nout;
                     }
                     if (prNode.needAtomic) {
                       atomicAdd(atomicVec[src], partial_sum);
                     } else {
                       nonAtomicVec[src] += partial_sum;
                     }
                   },
                   galois::no_stats(), galois::steal(),
                   galois::chunk_size<CHUNK_SIZE>(),
                   galois::loopname("computePartialPRContrib"));

    galois::do_all(galois::iterate(graph),
                   [&](const GNode& src) {
                     if (atomicVec[src] > nonAtomicVec[src]) {
                       assert(nonAtomicVec[prNode.src] == 0);
                       nonAtomicVec[src] = atomicVec[src];
                     }
                   },
                   galois::no_stats(), galois::loopname("AccumulateContrib"));

    // Finalize pagerank for this iteration
    galois::do_all(galois::iterate(graph),
                   [&](const GNode& src) {
                     constexpr const galois::MethodFlag flag =
                         galois::MethodFlag::UNPROTECTED;
                     LNode& sdata = graph.getData(src, flag);

                     // New value of pagerank after computing contributions from
                     // incoming edges in the original graph
                     float value = nonAtomicVec[src] * ALPHA + (1.0 - ALPHA);
                     // Find the delta in new and old pagerank values
                     float diff =
                         std::fabs(value - sdata.getPageRank(iteration));

                     // Do not update pagerank before the diff is computed since
                     // there is a
                     // data dependence on the pagerank value
                     sdata.setPageRank(iteration, value);
                     max_delta.update(diff);
                   },
                   galois::no_stats(), galois::loopname("PageRankFinalize"));

    float delta = max_delta.reduce();

#if DEBUG
    std::cout << "iteration: " << iteration << " max delta: " << delta << "\n";
#endif

    iteration++;
    if (delta <= tolerance || iteration >= maxIterations) {
      break;
    }
    max_delta.reset();

    // TODO: We can merge this loop to the earlier one, the downside is there
    // will be additional stores in case it was the last iteration. But maybe
    // the overhead of a Galois parallel loop is more than that?
    galois::do_all(galois::iterate(graph),
                   [&](const GNode& src) {
                     nonAtomicVec[src] = 0;
                     atomicVec[src]    = 0;
                   },
                   galois::no_stats(), galois::loopname("ClearVectors"));
  } // end while(true)

  if (iteration >= maxIterations) {
    std::cerr << "ERROR: failed to converge in " << iteration << " iterations"
              << std::endl;
  }

  if (iteration & 1) {
    // Result already in right place
  } else {
    finalizePR(graph);
  }
}
// FIXME: This is for debugging scalability and to avoid inlining.
struct PageRankPull {
  Graph& graph;
  unsigned int iteration;
  galois::GReduceMax<float>& max_delta;

  PageRankPull(Graph& _graph, unsigned int _it, galois::GReduceMax<float>& _mx)
      : graph(_graph), iteration(_it), max_delta(_mx) {}

  GALOIS_ATTRIBUTE_NOINLINE
  void static run(Graph& _graph) {
    unsigned int iteration = 0;
    galois::GReduceMax<float> max_delta;

    while (true) {
      galois::runtime::profileVtune(
          [&]() {
            galois::do_all(galois::iterate(_graph),
                           PageRankPull{_graph, iteration, max_delta},
                           galois::no_stats(), galois::steal(),
                           galois::chunk_size<CHUNK_SIZE>(),
                           galois::loopname("PageRank"));
          },
          "computePageRankProfileVTune");

      float delta = max_delta.reduce();

#if DEBUG
      std::cout << "iteration: " << iteration << " max delta: " << delta
                << "\n";
#endif

      iteration += 1;
      if (delta <= tolerance || iteration >= maxIterations) {
        break;
      }
      max_delta.reset();
    } // end while(true)

    if (iteration >= maxIterations) {
      std::cerr << "ERROR: failed to converge in " << iteration << " iterations"
                << std::endl;
    }

    if (iteration & 1) {
      // Result already in right place
    } else {
      finalizePR(_graph);
    }
  }

  GALOIS_ATTRIBUTE_NOINLINE
  void operator()(GNode src) const {
    constexpr const galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED;

    LNode& sdata = graph.getData(src, flag);
    float sum    = 0.0;

    for (auto jj = graph.edge_begin(src, flag), ej = graph.edge_end(src, flag);
         jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);

      LNode& ddata = graph.getData(dst, flag);
      sum += ddata.getPageRank(iteration) / ddata.nout;
    }

    // New value of pagerank after computing contributions from
    // incoming edges in the original graph
    float value = sum * ALPHA + (1.0 - ALPHA);
    // Find the delta in new and old pagerank values
    float diff = std::fabs(value - sdata.getPageRank(iteration));

    // Do not update pagerank before the diff is computed since
    // there is a data dependence on the pagerank value
    sdata.setPageRank(iteration, value);
    max_delta.update(diff);
  }
};

void computePageRank(Graph& graph) {
  unsigned int iteration = 0;
  galois::GReduceMax<float> max_delta;

  while (true) {

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          constexpr const galois::MethodFlag flag =
              galois::MethodFlag::UNPROTECTED;

          LNode& sdata = graph.getData(src, flag);
          float sum    = 0.0;

          for (auto jj = graph.edge_begin(src, flag),
                    ej = graph.edge_end(src, flag);
               jj != ej; ++jj) {
            GNode dst = graph.getEdgeDst(jj);

            LNode& ddata = graph.getData(dst, flag);
            sum += ddata.getPageRank(iteration) / ddata.nout;
          }

          // New value of pagerank after computing contributions from
          // incoming edges in the original graph
          float value = sum * ALPHA + (1.0 - ALPHA);
          // Find the delta in new and old pagerank values
          float diff = std::fabs(value - sdata.getPageRank(iteration));

          // Do not update pagerank before the diff is computed since
          // there is a data dependence on the pagerank value
          sdata.setPageRank(iteration, value);
          max_delta.update(diff);
        },
        galois::no_stats(), galois::steal(), galois::chunk_size<CHUNK_SIZE>(),
        galois::loopname("PageRank"));

    float delta = max_delta.reduce();

#if DEBUG
    std::cout << "iteration: " << iteration << " max delta: " << delta << "\n";
#endif

    iteration += 1;
    if (delta <= tolerance || iteration >= maxIterations) {
      break;
    }
    max_delta.reset();

  } // end while(true)

  if (iteration >= maxIterations) {
    std::cerr << "ERROR: failed to converge in " << iteration << " iterations"
              << std::endl;
  }

  if (iteration & 1) {
    // Result already in right place
  } else {
    finalizePR(graph);
  }
}

static void printPageRank(Graph& graph) {
  std::cout << "Id\tPageRank\n";
  int counter = 0;
  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ii++) {
    GNode src                    = *ii;
    Graph::node_data_reference n = graph.getData(src);
    std::cout << counter << " " << n.getPageRank() << "\n";
    counter++;
  }
}

template <typename Graph>
static void printTop(Graph& graph, int topn) {
  typedef typename Graph::node_data_reference node_data_reference;
  typedef TopPair<GNode> Pair;
  typedef std::map<Pair, GNode> Top;

  Top top;

  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src             = *ii;
    node_data_reference n = graph.getData(src);
    PRTy value            = n.getPageRank();
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

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer T("OverheadTime");
  T.start();

  Graph transposeGraph;
  std::cout << "Reading graph: " << filename << std::endl;
  galois::graphs::readGraph(transposeGraph, filename);
  std::cout << "Read " << transposeGraph.size() << " nodes, "
            << transposeGraph.sizeEdges() << " edges\n";

  galois::preAlloc(numThreads + (3 * transposeGraph.size() *
                                 sizeof(typename Graph::node_data_type)) /
                                    galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  std::cout << "Running synchronous Pull version, tolerance:" << tolerance
            << ", maxIterations:" << maxIterations << "\n";

  initNodeData(transposeGraph);
  computeOutDeg(transposeGraph);

  galois::StatTimer Tmain;
  Tmain.start();

  switch (algo) {
  case PR_Pull: {
    computePageRank(transposeGraph);
    break;
  }
  case PR_Pull_Profile: {
    PageRankPull::run(transposeGraph);
    break;
  }
  case PR_Pull_ET: {
    computePageRankET(transposeGraph);
  }
  }

  Tmain.stop();

  galois::reportPageAlloc("MeminfoPost");

  // if (!skipVerify) {
  //   printTop(transposeGraph, PRINT_TOP);
  // }

  printPageRank(transposeGraph);

  T.stop();

  return 0;
}
