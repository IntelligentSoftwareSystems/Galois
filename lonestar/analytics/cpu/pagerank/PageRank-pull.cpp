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
#include "galois/Galois.h"
#include "galois/LargeArray.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "galois/gstl.h"

const char* desc =
    "Computes page ranks a la Page and Brin. This is a pull-style algorithm.";

enum Algo { Topo = 0, Residual };

static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
                           cll::values(clEnumVal(Topo, "Topological"),
                                       clEnumVal(Residual, "Residual")),
                           cll::init(Residual));

//! Flag that forces user to be aware that they should be passing in a
//! transposed graph.
static cll::opt<bool>
    transposedGraph("transposedGraph",
                    cll::desc("Specify that the input graph is transposed"),
                    cll::init(false));

constexpr static const unsigned CHUNK_SIZE = 32;

struct LNode {
  PRTy value;
  uint32_t nout;
};

typedef galois::graphs::LC_CSR_Graph<LNode, void>::with_no_lockable<
    true>::type ::with_numa_alloc<true>::type Graph;
typedef typename Graph::GraphNode GNode;

using DeltaArray    = galois::LargeArray<PRTy>;
using ResidualArray = galois::LargeArray<PRTy>;

//! Initialize nodes for the topological algorithm.
void initNodeDataTopological(Graph& g) {
  galois::do_all(
      galois::iterate(g),
      [&](const GNode& n) {
        auto& sdata = g.getData(n, galois::MethodFlag::UNPROTECTED);
        sdata.value = INIT_RESIDUAL;
        sdata.nout  = 0;
      },
      galois::no_stats(), galois::loopname("initNodeData"));
}

//! Initialize nodes for the residual algorithm.
void initNodeDataResidual(Graph& g, DeltaArray& delta,
                          ResidualArray& residual) {
  galois::do_all(
      galois::iterate(g),
      [&](const GNode& n) {
        auto& sdata = g.getData(n, galois::MethodFlag::UNPROTECTED);
        sdata.value = 0;
        sdata.nout  = 0;
        delta[n]    = 0;
        residual[n] = INIT_RESIDUAL;
      },
      galois::no_stats(), galois::loopname("initNodeData"));
}

//! Computing outdegrees in the tranpose graph is equivalent to computing the
//! indegrees in the original graph.
void computeOutDeg(Graph& graph) {
  galois::StatTimer outDegreeTimer("computeOutDegFunc");
  outDegreeTimer.start();

  galois::LargeArray<std::atomic<size_t>> vec;
  vec.allocateInterleaved(graph.size());

  galois::do_all(
      galois::iterate(graph),
      [&](const GNode& src) { vec.constructAt(src, 0ul); }, galois::no_stats(),
      galois::loopname("InitDegVec"));

  galois::do_all(
      galois::iterate(graph),
      [&](const GNode& src) {
        for (auto nbr : graph.edges(src)) {
          GNode dst = graph.getEdgeDst(nbr);
          vec[dst].fetch_add(1ul);
        };
      },
      galois::steal(), galois::chunk_size<CHUNK_SIZE>(), galois::no_stats(),
      galois::loopname("computeOutDeg"));

  galois::do_all(
      galois::iterate(graph),
      [&](const GNode& src) {
        auto& srcData = graph.getData(src, galois::MethodFlag::UNPROTECTED);
        srcData.nout  = vec[src];
      },
      galois::no_stats(), galois::loopname("CopyDeg"));

  outDegreeTimer.stop();
}

/**
 * It does not calculate the pagerank for each iteration,
 * but only calculate the residual to be added from the previous pagerank to
 * the current one.
 * If the residual is smaller than the tolerance, that is not reflected to
 * the next pagerank.
 */
//! [scalarreduction]
void computePRResidual(Graph& graph, DeltaArray& delta,
                       ResidualArray& residual) {
  unsigned int iterations = 0;
  galois::GAccumulator<unsigned int> accum;

  while (true) {
    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          auto& sdata = graph.getData(src);
          delta[src]  = 0;

          //! Only the residual higher than tolerance will be reflected
          //! to the pagerank.
          if (residual[src] > tolerance) {
            PRTy oldResidual = residual[src];
            residual[src]    = 0.0;
            sdata.value += oldResidual;
            if (sdata.nout > 0) {
              delta[src] = oldResidual * ALPHA / sdata.nout;
              accum += 1;
            }
          }
        },
        galois::no_stats(), galois::loopname("PageRank_delta"));

    galois::do_all(
        galois::iterate(graph),
        [&](const GNode& src) {
          float sum = 0;
          for (auto nbr : graph.edges(src)) {
            GNode dst = graph.getEdgeDst(nbr);
            if (delta[dst] > 0) {
              sum += delta[dst];
            }
          }
          if (sum > 0) {
            residual[src] = sum;
          }
        },
        galois::steal(), galois::chunk_size<CHUNK_SIZE>(), galois::no_stats(),
        galois::loopname("PageRank"));

#if DEBUG
    std::cout << "iteration: " << iterations << "\n";
#endif
    iterations++;
    if (iterations >= maxIterations || !accum.reduce()) {
      break;
    }
    accum.reset();
  } ///< End while(true).
    //! [scalarreduction]

  if (iterations >= maxIterations) {
    std::cerr << "ERROR: failed to converge in " << iterations
              << " iterations\n";
  }
}

/**
 * PageRank pull topological.
 * Always calculate the new pagerank for each iteration.
 */
void computePRTopological(Graph& graph) {
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
            sum += ddata.value / ddata.nout;
          }

          //! New value of pagerank after computing contributions from
          //! incoming edges in the original graph.
          float value = sum * ALPHA + (1.0 - ALPHA);
          //! Find the delta in new and old pagerank values.
          float diff = std::fabs(value - sdata.value);

          //! Do not update pagerank before the diff is computed since
          //! there is a data dependence on the pagerank value.
          sdata.value = value;
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

  } ///< End while(true).

  if (iteration >= maxIterations) {
    std::cerr << "ERROR: failed to converge in " << iteration
              << " iterations\n";
  }
}

void prTopological(Graph& graph) {
  initNodeDataTopological(graph);
  computeOutDeg(graph);

  galois::StatTimer execTime("Timer_0");
  execTime.start();
  computePRTopological(graph);
  execTime.stop();
}

void prResidual(Graph& graph) {
  DeltaArray delta;
  delta.allocateInterleaved(graph.size());
  ResidualArray residual;
  residual.allocateInterleaved(graph.size());

  initNodeDataResidual(graph, delta, residual);
  computeOutDeg(graph);

  galois::StatTimer execTime("Timer_0");
  execTime.start();
  computePRResidual(graph, delta, residual);
  execTime.stop();
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url, &inputFile);

  if (!transposedGraph) {
    GALOIS_DIE("This application requires a transposed graph input;"
               " please use the -transposedGraph flag "
               " to indicate the input is a transposed graph.");
  }
  galois::StatTimer totalTime("TimerTotal");
  totalTime.start();

  Graph transposeGraph;
  std::cout << "WARNING: pull style algorithms work on the transpose of the "
               "actual graph\n"
            << "WARNING: this program assumes that " << inputFile
            << " contains transposed representation\n\n"
            << "Reading graph: " << inputFile << "\n";

  galois::graphs::readGraph(transposeGraph, inputFile);
  std::cout << "Read " << transposeGraph.size() << " nodes, "
            << transposeGraph.sizeEdges() << " edges\n";

  galois::preAlloc(2 * numThreads + (3 * transposeGraph.size() *
                                     sizeof(typename Graph::node_data_type)) /
                                        galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  switch (algo) {
  case Topo:
    std::cout << "Running Pull Topological version, tolerance:" << tolerance
              << ", maxIterations:" << maxIterations << "\n";
    prTopological(transposeGraph);
    break;
  case Residual:
    std::cout << "Running Pull Residual version, tolerance:" << tolerance
              << ", maxIterations:" << maxIterations << "\n";
    prResidual(transposeGraph);
    break;
  default:
    std::abort();
  }

  galois::reportPageAlloc("MeminfoPost");

  //! Sanity checking code.
  galois::GReduceMax<PRTy> maxRank;
  galois::GReduceMin<PRTy> minRank;
  galois::GAccumulator<PRTy> distanceSum;
  maxRank.reset();
  minRank.reset();
  distanceSum.reset();

  //! [example of no_stats]
  galois::do_all(
      galois::iterate(transposeGraph),
      [&](uint64_t i) {
        PRTy rank = transposeGraph.getData(i).value;

        maxRank.update(rank);
        minRank.update(rank);
        distanceSum += rank;
      },
      galois::loopname("Sanity check"), galois::no_stats());
  //! [example of no_stats]

  PRTy rMaxRank = maxRank.reduce();
  PRTy rMinRank = minRank.reduce();
  PRTy rSum     = distanceSum.reduce();
  galois::gInfo("Max rank is ", rMaxRank);
  galois::gInfo("Min rank is ", rMinRank);
  galois::gInfo("Sum is ", rSum);

  if (!skipVerify) {
    printTop(transposeGraph);
  }

#if DEBUG
  printPageRank(transposeGraph);
#endif

  totalTime.stop();

  return 0;
}
