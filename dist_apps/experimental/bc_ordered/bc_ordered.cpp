/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
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

constexpr static const char* const REGION_NAME = "BC_LEVEL";

#include <iostream>
#include <limits>
#include <random>

#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"
#include "galois/DReducible.h"
#include "galois/runtime/Tracer.h"

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/
namespace cll = llvm::cl;
static cll::opt<std::string> sourcesToUse("sourcesToUse",
                                          cll::desc("Whitespace separated list "
                                                    "of sources in a file to "
                                                    "use in BC"),
                                          cll::init(""));
static cll::opt<unsigned int>
    maxIterations("maxIterations",
                  cll::desc("Maximum iterations: Default 10000"),
                  cll::init(10000));
static cll::opt<bool> singleSourceBC("singleSource",
                                     cll::desc("Use for single source BC"),
                                     cll::init(false));
static cll::opt<unsigned long long>
    startSource("startNode", // not uint64_t due to a bug in llvm cl
                cll::desc("Starting source node used for "
                          "betweeness-centrality"),
                cll::init(0));
static cll::opt<unsigned int>
    numberOfSources("numOfSources",
                    cll::desc("Number of sources to use for "
                              "betweeness-centraility"),
                    cll::init(0));
static cll::opt<bool> randomSources("randomSources",
                                    cll::desc("Use random sources."),
                                    cll::init(false));

/******************************************************************************/
/* Graph structure declarations */
/******************************************************************************/
static uint64_t current_src_node = 0;
const uint32_t infinity          = std::numeric_limits<uint32_t>::max();

// NOTE: types assume that fields will not reach uint64_t: it may
// need to be changed for very large graphs
struct NodeData {
  // SSSP vars
  std::atomic<uint32_t> current_length;

  uint32_t old_length;

  // Betweeness centrality vars
  uint64_t num_shortest_paths; // 64 because # paths can get really big
  uint32_t num_successors;
  std::atomic<uint32_t> num_predecessors;
  std::atomic<uint32_t> trim;
  std::atomic<uint64_t> to_add;

  float to_add_float;
  float dependency;

  float betweeness_centrality;

  // used to determine if data has been propagated yet
  uint8_t propagation_flag;

#ifdef BCDEBUG
  void dump() {
    galois::gPrint("DUMP: ", current_length.load(), " ", old_length, " ",
                   num_shortest_paths, " ", num_successors, " ",
                   num_predecessors.load(), " ", trim.load(), " ",
                   to_add.load(), " ", to_add_float, " ", dependency, " ",
                   (bool)propagation_flag, "\n");
  }
#endif
};

static std::set<uint64_t> random_sources = std::set<uint64_t>();
// reading in list of sources to operate on if provided
std::ifstream sourceFile;
std::vector<uint64_t> sourceVector;

#ifndef __USE_BFS__
typedef galois::graphs::DistGraph<NodeData, unsigned int> Graph;
#else
typedef galois::graphs::DistGraph<NodeData, void> Graph;
#endif

typedef typename Graph::GraphNode GNode;

// bitsets for tracking updates
galois::DynamicBitSet bitset_to_add;
galois::DynamicBitSet bitset_to_add_float;
galois::DynamicBitSet bitset_num_shortest_paths;
galois::DynamicBitSet bitset_num_successors;
galois::DynamicBitSet bitset_num_predecessors;
galois::DynamicBitSet bitset_trim;
galois::DynamicBitSet bitset_current_length;
galois::DynamicBitSet bitset_propagation_flag;
galois::DynamicBitSet bitset_dependency;

// sync structures
#include "bc_ordered_sync.hh"

/******************************************************************************/
/* Functors for running the algorithm */
/******************************************************************************/

/**
 * Functor to initialize fields on the nodes to starting value.
 */
struct InitializeGraph {
  Graph* graph;

  InitializeGraph(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph) {
    galois::do_all(galois::iterate(allNodes.begin(), allNodes.end()),
                   InitializeGraph{&_graph}, galois::no_stats(),
                   galois::loopname("InitializeGraph"));
  }

  void operator()(GNode node) const {
    NodeData& node_data = graph->getData(node);

    // TODO
  }
};

struct ForwardPhase {

}

// TODO

/******************************************************************************/
/* Sanity check */
/******************************************************************************/

struct Sanity {
  Graph* graph;

  galois::DGReduceMax<float>& DGAccumulator_max;
  galois::DGReduceMin<float>& DGAccumulator_min;
  galois::DGAccumulator<float>& DGAccumulator_sum;

  Sanity(Graph* _graph, galois::DGReduceMax<float>& _DGAccumulator_max,
         galois::DGReduceMin<float>& _DGAccumulator_min,
         galois::DGAccumulator<float>& _DGAccumulator_sum)
      : graph(_graph), DGAccumulator_max(_DGAccumulator_max),
        DGAccumulator_min(_DGAccumulator_min),
        DGAccumulator_sum(_DGAccumulator_sum) {}

  void static go(Graph& _graph, galois::DGReduceMax<float>& DGA_max,
                 galois::DGReduceMin<float>& DGA_min,
                 galois::DGAccumulator<float>& DGA_sum) {
    DGA_max.reset();
    DGA_min.reset();
    DGA_sum.reset();

    galois::do_all(galois::iterate(_graph.masterNodesRange().begin(),
                                   _graph.masterNodesRange().end()),
                   Sanity(&_graph, DGA_max, DGA_min, DGA_sum),
                   galois::no_stats(), galois::loopname("Sanity"));

    float max_bc = DGA_max.reduce();
    float min_bc = DGA_min.reduce();
    float bc_sum = DGA_sum.reduce();

    // Only node 0 will print data
    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
      galois::gPrint("Max BC is ", max_bc, "\n");
      galois::gPrint("Min BC is ", min_bc, "\n");
      galois::gPrint("BC sum is ", bc_sum, "\n");
    }
  }

  /* Gets the max, min rank from all owned nodes and
   * also the sum of ranks */
  void operator()(GNode src) const {
    NodeData& sdata = graph->getData(src);

    DGAccumulator_max.update(sdata.betweeness_centrality);
    DGAccumulator_min.update(sdata.betweeness_centrality);
    DGAccumulator_sum += sdata.betweeness_centrality;
  }
};

/******************************************************************************/
/* Main method for running */
/******************************************************************************/

constexpr static const char* const name =
    "Level by Level Betweeness Centrality - "
    "Distributed Heterogeneous.";
constexpr static const char* const desc = "Level by Level Betweeness Centrality"
                                          " on Distributed Galois.";
constexpr static const char* const url = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  auto& net = galois::runtime::getSystemNetworkInterface();
  if (net.ID == 0) {
    galois::runtime::reportParam(REGION_NAME, "Max Iterations",
                                 (unsigned long)maxIterations);
  }

  galois::StatTimer StatTimer_total("TimerTotal", REGION_NAME);

  StatTimer_total.start();

  Graph* h_graph = distGraphInitialization<NodeData, void>();

  if (!randomSources) {
    for (unsigned i = 0; i < numberOfSources; i++) {
      random_sources.insert(i);
    }
  } else {
    // random num generate for sources
    std::minstd_rand0 r_generator;
    r_generator.seed(100);
    std::uniform_int_distribution<uint64_t> r_dist(0,
                                                   h_graph->globalSize() - 1);

    if (numberOfSources != 0) {
      // uncomment this to have srcnodeid included as well
      // random_sources.insert(startSource);

      while (random_sources.size() < numberOfSources) {
        random_sources.insert(r_dist(r_generator));
      }
    }
  }

  if (sourcesToUse != "") {
    sourceFile.open(sourcesToUse);
    std::vector<uint64_t> t(std::istream_iterator<uint64_t>{sourceFile},
                            std::istream_iterator<uint64_t>{});
    sourceVector = t;
    sourceFile.close();
  }

#ifndef NDEBUG
  int counter = 0;
  for (auto i = random_sources.begin(); i != random_sources.end(); i++) {
    printf("Source #%d: %lu\n", counter, *i);
    counter++;
  }
#endif

  bitset_to_add.resize(h_graph->size());
  bitset_to_add_float.resize(h_graph->size());
  bitset_num_shortest_paths.resize(h_graph->size());
  bitset_num_successors.resize(h_graph->size());
  bitset_num_predecessors.resize(h_graph->size());
  bitset_trim.resize(h_graph->size());
  bitset_current_length.resize(h_graph->size());
  bitset_propagation_flag.resize(h_graph->size());
  bitset_dependency.resize(h_graph->size());

  galois::gPrint("[", net.ID, "] InitializeGraph::go called\n");

  galois::StatTimer StatTimer_graph_init("TIMER_GRAPH_INIT", REGION_NAME);
  StatTimer_graph_init.start();
  InitializeGraph::go((*h_graph));
  StatTimer_graph_init.stop();
  galois::runtime::getHostBarrier().wait();

  // shared DG accumulator among all steps
  galois::DGAccumulator<uint32_t> dga;

  // sanity dg accumulators
  galois::DGReduceMax<float> dga_max;
  galois::DGReduceMin<float> dga_min;
  galois::DGAccumulator<float> dga_sum;

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] BC::go run ", run, " called\n");
    std::string timer_str("Timer_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    StatTimer_main.start();
    BC::go(*h_graph, dga);
    StatTimer_main.stop();

    Sanity::go(*h_graph, dga_max, dga_min, dga_sum);

    // re-init graph for next run
    if ((run + 1) != numRuns) {
      galois::runtime::getHostBarrier().wait();
      (*h_graph).set_num_run(run + 1);

      bitset_to_add.reset();
      bitset_to_add_float.reset();
      bitset_num_shortest_paths.reset();
      bitset_num_successors.reset();
      bitset_num_predecessors.reset();
      bitset_trim.reset();
      bitset_current_length.reset();
      bitset_propagation_flag.reset();
      bitset_dependency.reset();

      InitializeGraph::go((*h_graph));
      galois::runtime::getHostBarrier().wait();
    }
  }

  StatTimer_total.stop();

  // Verify, i.e. print out graph data for examination
  if (verify) {
    char* v_out = (char*)malloc(40);

    for (auto ii = (*h_graph).masterNodesRange().begin();
         ii != (*h_graph).masterNodesRange().end(); ++ii) {
      // outputs betweenness centrality
      sprintf(v_out, "%lu %.9f\n", (*h_graph).getGID(*ii),
              (*h_graph).getData(*ii).betweeness_centrality);
      galois::runtime::printOutput(v_out);
    }

    free(v_out);
  }

  return 0;
}
