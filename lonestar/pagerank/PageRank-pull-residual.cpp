/** Residual based Page Rank -*- C++ -*-
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
 * Compute pageRank Pull version using residual on distributed Galois.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 * @author Roshan Dathathri <roshan@cs.utexas.edu>
 * @author Loc Hoang <l_hoang@utexas.edu> (sanity check operators)
 */

#include "Lonestar/BoilerPlate.h"
#include "PageRank-constants.h"
#include "galois/Galois.h"
#include "galois/LargeArray.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "galois/gstl.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

namespace cll = llvm::cl;
const char* desc =
    "Computes page ranks a la Page and Brin. This is a push-style algorithm.";

// We require a transpose graph since this is a pull-style algorithm
static cll::opt<std::string> filename(cll::Positional,
                                      cll::desc("<tranpose of input graph>"),
                                      cll::Required);
static cll::opt<float> tolerance("tolerance",
                                 cll::desc("tolerance for residual"),
                                 cll::init(TOLERANCE));
static cll::opt<unsigned int>
    maxIterations("maxIterations",
                  cll::desc("Maximum iterations: Default 1000"),
                  cll::init(MAX_ITER));

struct LNode {
  PRTy value;
  std::atomic<uint32_t> nout;
};

galois::LargeArray<PRTy> delta;
galois::LargeArray<PRTy> residual;

typedef galois::graphs::LC_CSR_Graph<LNode, void>::with_no_lockable<
    true>::type ::with_numa_alloc<true>::type Graph;
typedef typename Graph::GraphNode GNode;

/* (Re)initialize all fields to 0 except for residual which needs to be 0.15
 * everywhere */
struct ResetGraph {
  Graph* graph;

  ResetGraph(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    galois::do_all(galois::iterate(_graph), ResetGraph{&_graph},
                   galois::no_stats(), galois::loopname("ResetGraph"));
  }

  void operator()(GNode src) const {
    auto& sdata   = graph->getData(src);
    sdata.value   = 0;
    sdata.nout    = 0;
    delta[src]    = 0;
    residual[src] = ALPHA;
  }
};

struct InitializeGraph {
  Graph* graph;

  InitializeGraph(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    ResetGraph::go(_graph);

    // doing a local do all because we are looping over edges
    galois::do_all(galois::iterate(_graph), InitializeGraph{&_graph},
                   galois::steal(), galois::no_stats(),
                   galois::loopname("InitializeGraph"));
  }

  // Calculate "outgoing" edges for destination nodes (note we are using
  // the tranpose graph for pull algorithms)
  void operator()(GNode src) const {
    for (auto nbr : graph->edges(src)) {
      GNode dst   = graph->getEdgeDst(nbr);
      auto& ddata = graph->getData(dst);
      galois::atomicAdd(ddata.nout, (uint32_t)1);
    }
  }
};

struct PageRank_delta {
  const float& local_alpha;
  cll::opt<float>& local_tolerance;
  Graph* graph;

  galois::GAccumulator<unsigned int>& GAccumulator_accum;

  PageRank_delta(const float& _local_alpha, cll::opt<float>& _local_tolerance,
                 Graph* _graph, galois::GAccumulator<unsigned int>& _dga)
      : local_alpha(_local_alpha), local_tolerance(_local_tolerance),
        graph(_graph), GAccumulator_accum(_dga) {}

  void static go(Graph& _graph, galois::GAccumulator<unsigned int>& dga) {
    const auto& allNodes = _graph.allNodesRange();

    galois::do_all(galois::iterate(_graph),
                   PageRank_delta{ALPHA, tolerance, &_graph, dga},
                   galois::no_stats(), galois::loopname("PageRank_delta"));
  }

  void operator()(GNode src) const {
    auto& sdata = graph->getData(src);
    delta[src]  = 0;

    if (residual[src] > local_tolerance) {
      sdata.value += residual[src];
      if (sdata.nout > 0) {
        delta[src] = residual[src] * (1 - local_alpha) / sdata.nout;
        GAccumulator_accum += 1;
      }
      residual[src] = 0;
    }
  }
};

struct PageRank {
  Graph* graph;

  PageRank(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph, galois::GAccumulator<unsigned int>& dga) {
    unsigned _num_iterations = 0;

    do {
      _graph.set_num_iter(_num_iterations);
      dga.reset();
      PageRank_delta::go(_graph, dga);

      galois::do_all(
          galois::iterate(_graph), PageRank{&_graph}, galois::steal(),
          galois::no_stats(),
          galois::loopname(_graph.get_run_identifier("PageRank").c_str()));

      // galois::runtime::reportStat_Tsum(
      //     REGION_NAME, "NUM_WORK_ITEMS_" + (_graph.get_run_identifier()),
      //     (unsigned long)dga.read_local());

      ++_num_iterations;
    } while ((_num_iterations < maxIterations) &&
             dga.reduce(_graph.get_run_identifier()));

    // galois::runtime::reportStat_Single(
    //     REGION_NAME, "NUM_ITERATIONS_" +
    //     std::to_string(_graph.get_run_num()), (unsigned
    //     long)_num_iterations);
  }

  // Pull deltas from neighbor nodes, then add to self-residual
  void operator()(GNode src) const {
    float sum = 0;
    for (auto nbr : graph->edges(src)) {
      GNode dst = graph->getEdgeDst(nbr);
      if (delta[dst] > 0) {
        sum += delta[dst];
      }
    }
    if (sum > 0) {
      galois::add(residual[src], sum);
    }
  }
};

// Gets various values from the pageranks values/residuals of the graph
// struct PageRankSanity {
//   cll::opt<float>& local_tolerance;
//   Graph* graph;

//   galois::GAccumulator<float>& GAccumulator_sum;
//   galois::GAccumulator<float>& GAccumulator_sum_residual;
//   galois::GAccumulator<uint64_t>& GAccumulator_residual_over_tolerance;

//   galois::GReduceMax<float>& max_value;
//   galois::GReduceMin<float>& min_value;
//   galois::GReduceMax<float>& max_residual;
//   galois::GReduceMin<float>& min_residual;

//   PageRankSanity(
//       cll::opt<float>& _local_tolerance, Graph* _graph,
//       galois::GAccumulator<float>& _GAccumulator_sum,
//       galois::GAccumulator<float>& _GAccumulator_sum_residual,
//       galois::GAccumulator<uint64_t>& _GAccumulator_residual_over_tolerance,
//       galois::GReduceMax<float>& _max_value,
//       galois::GReduceMin<float>& _min_value,
//       galois::GReduceMax<float>& _max_residual,
//       galois::GReduceMin<float>& _min_residual)
//       : local_tolerance(_local_tolerance), graph(_graph),
//         GAccumulator_sum(_GAccumulator_sum),
//         GAccumulator_sum_residual(_GAccumulator_sum_residual),
//         GAccumulator_residual_over_tolerance(
//             _GAccumulator_residual_over_tolerance),
//         max_value(_max_value), min_value(_min_value),
//         max_residual(_max_residual), min_residual(_min_residual) {}

//   void static go(Graph& _graph, galois::GAccumulator<float>& DGA_sum,
//                  galois::GAccumulator<float>& DGA_sum_residual,
//                  galois::GAccumulator<uint64_t>& DGA_residual_over_tolerance,
//                  galois::GReduceMax<float>& max_value,
//                  galois::GReduceMin<float>& min_value,
//                  galois::GReduceMax<float>& max_residual,
//                  galois::GReduceMin<float>& min_residual) {
//     DGA_sum.reset();
//     DGA_sum_residual.reset();
//     max_value.reset();
//     max_residual.reset();
//     min_value.reset();
//     min_residual.reset();
//     DGA_residual_over_tolerance.reset();

//     {
//       galois::do_all(galois::iterate(_graph.masterNodesRange().begin(),
//                                      _graph.masterNodesRange().end()),
//                      PageRankSanity(tolerance, &_graph, DGA_sum,
//                                     DGA_sum_residual,
//                                     DGA_residual_over_tolerance, max_value,
//                                     min_value, max_residual, min_residual),
//                      galois::no_stats(), galois::loopname("PageRankSanity"));
//     }

//     float max_rank          = max_value.reduce();
//     float min_rank          = min_value.reduce();
//     float rank_sum          = DGA_sum.reduce();
//     float residual_sum      = DGA_sum_residual.reduce();
//     uint64_t over_tolerance = DGA_residual_over_tolerance.reduce();
//     float max_res           = max_residual.reduce();
//     float min_res           = min_residual.reduce();

//     galois::gPrint("Max rank is ", max_rank, "\n");
//     galois::gPrint("Min rank is ", min_rank, "\n");
//     galois::gPrint("Rank sum is ", rank_sum, "\n");
//     galois::gPrint("Residual sum is ", residual_sum, "\n");
//     galois::gPrint("# nodes with residual over ", tolerance, " (tolerance) is
//     ",
//                    over_tolerance, "\n");
//     galois::gPrint("Max residual is ", max_res, "\n");
//     galois::gPrint("Min residual is ", min_res, "\n");
//   }

//   /* Gets the max, min rank from all owned nodes and
//    * also the sum of ranks */
//   void operator()(GNode src) const {
//     NodeData& sdata = graph->getData(src);

//     max_value.update(sdata.value);
//     min_value.update(sdata.value);
//     max_residual.update(residual[src]);
//     min_residual.update(residual[src]);

//     GAccumulator_sum += sdata.value;
//     GAccumulator_sum_residual += residual[src];

//     if (residual[src] > local_tolerance) {
//       GAccumulator_residual_over_tolerance += 1;
//     }
//   }
// };

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer overheadTime("OverheadTime");
  overheadTime.start();

  Graph transposeGraph;
  galois::graphs::readGraph(transposeGraph, filename);
  std::cout << "Reading graph: " << filename << std::endl;
  std::cout << "Read " << transposeGraph.size() << " nodes, "
            << transposeGraph.sizeEdges() << " edges\n";

  residual.allocateInterleaved(transposeGraph.size());
  delta.allocateInterleaved(transposeGraph.size());

  galois::preAlloc(numThreads + (3 * transposeGraph.size() *
                                 sizeof(typename Graph::node_data_type)) /
                                    galois::runtime::pagePoolSize());
  galois::reportPageAlloc("MeminfoPre");

  std::cout << "Running synchronous Pull version, tolerance:" << tolerance
            << ", maxIterations:" << maxIterations << "\n";

  galois::StatTimer initGraph("InitGraph");
  initGraph.start();
  InitializeGraph::go(*hg);
  initGraph.stop();

  galois::GAccumulator<unsigned int> PageRank_accum;

  galois::GAccumulator<float> DGA_sum;
  galois::GAccumulator<float> DGA_sum_residual;
  galois::GAccumulator<uint64_t> DGA_residual_over_tolerance;
  galois::GReduceMax<float> max_value;
  galois::GReduceMin<float> min_value;
  galois::GReduceMax<float> max_residual;
  galois::GReduceMin<float> min_residual;

  galois::StatTimer prTimer("PageRank");
  prTimer.start();
  PageRank::go(*hg, PageRank_accum);
  prTimer.stop();

  // // sanity check
  // PageRankSanity::go(*hg, DGA_sum, DGA_sum_residual,
  //                    DGA_residual_over_tolerance, max_value, min_value,
  //                    max_residual, min_residual);

  InitializeGraph::go(*hg);

  overheadTime.stop();

  // // Verify
  // if (verify) {
  //   for (auto ii = (*hg).masterNodesRange().begin();
  //        ii != (*hg).masterNodesRange().end(); ++ii) {
  //     galois::runtime::printOutput("% %\n", (*hg).getGID(*ii),
  //                                  (*hg).getData(*ii).value);
  //   }
  // }

  return 0;
}
