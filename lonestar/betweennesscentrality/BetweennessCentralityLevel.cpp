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

/**
 * This version of BC-Level uses an option in the synchronization runtime to
 * avoid the overheads of having 2 extra accumulator variables.
 */

//#define BCDEBUG

constexpr static const char* const REGION_NAME = "BC";

#include <iostream>
#include <limits>
#include <fstream>

#include "galois/gstl.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/DynamicBitset.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"
#include "galois/AtomicHelpers.h"

#include "Lonestar/BoilerPlate.h"

#include "galois/runtime/Profile.h"



// type of the num shortest paths variable
using ShortPathType = double;

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/
namespace cll = llvm::cl;
static cll::opt<std::string>
    filename(cll::Positional, cll::desc("<input graph>"), cll::Required);
static cll::opt<std::string>
    sourcesToUse("sourcesToUse",
                 cll::desc("Whitespace separated list "
                           "of sources in a file to "
                           "use in BC (default empty)"),
                 cll::init(""));
static cll::opt<bool>
    singleSourceBC("singleSource",
                   cll::desc("Use for single source BC (default off)"),
                   cll::init(false));
static cll::opt<unsigned long long>
    startSource("startNode", // not uint64_t due to a bug in llvm cl
                cll::desc("Starting source node used for "
                          "betweeness-centrality (default 0)"),
                cll::init(0));
static cll::opt<unsigned int>
    numberOfSources("numOfSources",
                    cll::desc("Number of sources to use for "
                              "betweeness-centraility (default all)"),
                    cll::init(0));
static cll::opt<unsigned int>
    numRuns("numRuns",
              cll::desc("Number of runs (default value 1)"),
              cll::init(1));
static cll::opt<bool>
    verify("verify",
              cll::desc("Flag to verify(default: false)"),
              cll::init(false));


/******************************************************************************/
/* Graph structure declarations */
/******************************************************************************/
const uint32_t infinity          = std::numeric_limits<uint32_t>::max() / 4;
static uint64_t current_src_node = 0;
// global round numbers; 1 for forward, 1 for back; used in sync structs as well
uint32_t globalRoundNumber = 0;
uint32_t backRoundCount = 0;

// NOTE: types assume that these values will not reach uint64_t: it may
// need to be changed for very large graphs
struct NodeData {
  // SSSP vars
  std::atomic<uint32_t> current_length;
  // Betweeness centrality vars
  std::atomic<ShortPathType> num_shortest_paths;
  float dependency;
  float betweeness_centrality;

//#ifdef BCDEBUG
  void dump() {
    galois::gPrint("DUMP: ", current_length.load(), " ",
                   num_shortest_paths.load(), " ", dependency, "\n");
  }
//#endif
};

// reading in list of sources to operate on if provided
std::ifstream sourceFile;
std::vector<uint64_t> sourceVector;

using Graph =
    galois::graphs::LC_CSR_Graph<NodeData, void>::with_no_lockable<true>::type::with_numa_alloc<true>::type;
using GNode = Graph::GraphNode;


//using Graph = galois::graphs::DistGraph<NodeData, void>;
//using GNode = typename Graph::GraphNode;

// bitsets for tracking updates
galois::DynamicBitSet bitset_num_shortest_paths;
galois::DynamicBitSet bitset_current_length;
galois::DynamicBitSet bitset_dependency;

// sync structures
//#include "bc_level_sync.hh"

/******************************************************************************/
/* Functors for running the algorithm */
/******************************************************************************/
struct InitializeGraph {
  Graph* graph;

  InitializeGraph(Graph* _graph) : graph(_graph) {}

  /* Initialize the graph */
  void static go(Graph& _graph) {

      galois::do_all(
          galois::iterate(_graph),
          InitializeGraph{&_graph}, galois::no_stats(),
          galois::loopname("InitializeGraph"));
  }

  /* Functor passed into the Galois operator to carry out initialization;
   * reset everything */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    src_data.betweeness_centrality = 0;
    src_data.num_shortest_paths    = 0;
    src_data.dependency            = 0;
  }
};

/* This is used to reset node data when switching to a difference source */
struct InitializeIteration {
  const uint32_t& local_infinity;
  const uint64_t& local_current_src_node;
  Graph* graph;

  InitializeIteration(const uint32_t& _local_infinity,
                      const uint64_t& _local_current_src_node, Graph* _graph)
      : local_infinity(_local_infinity),
        local_current_src_node(_local_current_src_node), graph(_graph) {}

  /* Reset necessary graph metadata for next iteration of SSSP */
  void static go(Graph& _graph) {

      galois::do_all(
          galois::iterate(_graph),
          InitializeIteration{infinity, current_src_node, &_graph},
          galois::loopname("InitializeIteration"),
          galois::no_stats());
  }

  /* Functor passed into the Galois operator to carry out reset of node data
   * (aside from betweeness centrality measure */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    bool is_source = (src == local_current_src_node);

    if (!is_source) {
      src_data.current_length     = local_infinity;
      src_data.num_shortest_paths = 0;
    } else {
      src_data.current_length     = 0;
      src_data.num_shortest_paths = 1;
    }
    src_data.dependency       = 0;
  }
};

/**
 * Forward pass does level by level BFS to find distances and number of
 * shortest paths
 */
struct ForwardPass {
  Graph* graph;
  galois::GAccumulator<uint32_t>& dga;
  uint32_t r;

  ForwardPass(Graph* _graph, galois::GAccumulator<uint32_t>& _dga, 
              uint32_t roundNum)
    : graph(_graph), dga(_dga), r(roundNum) {}

  /**
   * Level by level BFS while also finding number of shortest paths to a
   * particular node in the BFS tree.
   *
   * @param _graph Graph to use
   * @param _dga distributed accumulator
   * @param[out] roundNumber Number of rounds taken to finish
   */
  void static go(Graph& _graph, galois::GAccumulator<uint32_t>& _dga) {
    globalRoundNumber = 0;

    do {
      _dga.reset();

      galois::do_all(
        galois::iterate(_graph),
        ForwardPass(&_graph, _dga, globalRoundNumber),
        galois::loopname("ForwardPass"),
        galois::steal(),
        galois::no_stats()
      );

      globalRoundNumber++;
    } while (_dga.reduce());
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length == r) {
      for (auto current_edge : graph->edges(src)) {
        GNode dst = graph->getEdgeDst(current_edge);
        auto& dst_data = graph->getData(dst);
        uint32_t new_dist = 1 + src_data.current_length;
        uint32_t old = galois::atomicMin(dst_data.current_length, new_dist);

        if (old > new_dist) {
          //assert(dst_data.current_length == r + 1);
          //assert(src_data.num_shortest_paths > 0);

          bitset_current_length.set(dst);
          galois::atomicAdd(dst_data.num_shortest_paths, 
                            src_data.num_shortest_paths.load());
          bitset_num_shortest_paths.set(dst);

          dga += 1;
        } else if (old == new_dist) {
          //assert(src_data.num_shortest_paths > 0);
          //assert(dst_data.current_length == r + 1);

          galois::atomicAdd(dst_data.num_shortest_paths, 
                            src_data.num_shortest_paths.load());
          bitset_num_shortest_paths.set(dst);

          dga += 1;
        }
      }
    }
  }
};


#if 0
/**
 * Synchronize num shortest paths on destinations (should already
 * exist on all sources).
 */
struct MiddleSync {
  Graph* graph;
  const uint32_t local_infinity;

  MiddleSync(Graph* _graph, const uint32_t li) 
    : graph(_graph), local_infinity(li) {};
           
  void static go(Graph& _graph, const uint32_t _li) {
    // step only required if more than one host
    if (galois::runtime::getSystemNetworkInterface().Num > 1) {
      const auto& masters = _graph.masterNodesRange();

      galois::do_all(
        galois::iterate(masters.begin(), masters.end()),
        MiddleSync(&_graph, _li),
        galois::loopname(_graph.get_run_identifier("MiddleSync").c_str()),
        galois::no_stats()
      );

      _graph.sync<writeSource, readAny, Reduce_set_num_shortest_paths,
                  Broadcast_num_shortest_paths>("MiddleSync");
    }
  }

  /**
   * Set node for sync if it has a non-zero distance
   */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length != local_infinity) {
      bitset_num_shortest_paths.set(src);
    }
  }
};
#endif

/**
 * Propagate dependency backward by iterating backward over levels of BFS tree
 */
struct BackwardPass {
  Graph* graph;
  uint32_t r;

  BackwardPass(Graph* _graph, uint32_t roundNum) : graph(_graph), r(roundNum) {}

  void static go(Graph& _graph, uint32_t roundNumber) {

    backRoundCount = roundNumber - 1;

    for (; backRoundCount > 0; backRoundCount--) {
      galois::do_all(
        galois::iterate(_graph),
        BackwardPass(&_graph, backRoundCount),
        galois::loopname("BackwardPass"),
        galois::steal(),
        galois::no_stats()
      );

    }
  }

  /**
   * If on the correct level, calculate self-depndency by checking successor
   * nodes.
   */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.current_length == r) {
      uint32_t dest_to_find = src_data.current_length + 1;
      for (auto current_edge : graph->edges(src)) {
        GNode dst = graph->getEdgeDst(current_edge);
        auto& dst_data = graph->getData(dst);

        if (dest_to_find == dst_data.current_length.load()) {
          float contrib = ((float)1 + dst_data.dependency) /
                          dst_data.num_shortest_paths;
          src_data.dependency = src_data.dependency + contrib;
          bitset_dependency.set(src);
        }
      }
      src_data.dependency *= src_data.num_shortest_paths;
    }
  }
};


struct BC {
  Graph* graph;

  BC(Graph* _graph) : graph(_graph) {}

  void static go(Graph& _graph, galois::GAccumulator<uint32_t>& dga) {
    globalRoundNumber = 0;
    // reset the graph aside from the between-cent measure
    InitializeIteration::go(_graph);
    // get distances and num paths
    ForwardPass::go(_graph, dga);

    // dependency calc only matters if there's a node with distance at
    // least 2
    if (globalRoundNumber > 2) {
      //MiddleSync::go(_graph, infinity);
      BackwardPass::go(_graph, globalRoundNumber - 1);

      // finally, since dependencies are finalized for this round at this
      // point, add them to the betweeness centrality measure on each node
      galois::do_all(
        galois::iterate(_graph),
        BC(&_graph),
        galois::no_stats(),
        galois::loopname("BC")
      );
    }
  }

  /**
   * Adds dependency measure to BC measure 
   */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    if (src_data.dependency > 0) {
      src_data.betweeness_centrality += src_data.dependency;
    }
  }
};

/******************************************************************************/
/* Sanity check */
/******************************************************************************/

struct Sanity {
  Graph* graph;

  galois::GReduceMax<float>& DGAccumulator_max;
  galois::GReduceMin<float>& DGAccumulator_min;
  galois::GAccumulator<float>& DGAccumulator_sum;

  Sanity(Graph* _graph, galois::GReduceMax<float>& _DGAccumulator_max,
         galois::GReduceMin<float>& _DGAccumulator_min,
         galois::GAccumulator<float>& _DGAccumulator_sum)
      : graph(_graph), DGAccumulator_max(_DGAccumulator_max),
        DGAccumulator_min(_DGAccumulator_min),
        DGAccumulator_sum(_DGAccumulator_sum) {}

  void static go(Graph& _graph, galois::GReduceMax<float>& DGA_max,
                 galois::GReduceMin<float>& DGA_min,
                 galois::GAccumulator<float>& DGA_sum) {

    DGA_max.reset();
    DGA_min.reset();
    DGA_sum.reset();

    galois::do_all(galois::iterate(_graph),
                   Sanity(&_graph, DGA_max, DGA_min, DGA_sum),
                   galois::no_stats(), galois::loopname("Sanity"));

    float max_bc = DGA_max.reduce();
    float min_bc = DGA_min.reduce();
    float bc_sum = DGA_sum.reduce();

    // Only node 0 will print data
      galois::gPrint("Max BC is ", max_bc, "\n");
      galois::gPrint("Min BC is ", min_bc, "\n");
      galois::gPrint("BC sum is ", bc_sum, "\n");
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

constexpr static const char* const name = "Betweeness Centrality Level by Level";
constexpr static const char* const desc =
    "Betweeness Centrality Level by Level on Distributed Galois.";
constexpr static const char* const url = 0;

int main(int argc, char** argv) {

  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, NULL);
  Graph graph;

  galois::StatTimer StatTimer_total("TimerTotal", REGION_NAME);

  StatTimer_total.start();

  galois::StatTimer StatTimer_graphConstuct("TimerConstructGraph", "BFS");
  std::cout << "1 : Reading from file: " << filename << std::endl;
  StatTimer_graphConstuct.start();
  std::cout << "2 : Calling allocateAndLoadGraph : " << filename << std::endl;
  galois::graphs::readGraph(graph, filename);

  if (sourcesToUse != "") {
    sourceFile.open(sourcesToUse);
    std::vector<uint64_t> t(std::istream_iterator<uint64_t>{sourceFile},
                            std::istream_iterator<uint64_t>{});
    sourceVector = t;
    sourceFile.close();
  }

  bitset_num_shortest_paths.resize(graph.size());
  bitset_current_length.resize(graph.size());
  bitset_dependency.resize(graph.size());

  galois::gPrint(" InitializeGraph::go called\n");

  InitializeGraph::go((graph));

  // shared DG accumulator among all steps
  galois::GAccumulator<uint32_t> dga;

  // sanity dg accumulators
  galois::GReduceMax<float> dga_max;
  galois::GReduceMin<float> dga_min;
  galois::GAccumulator<float> dga_sum;

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint(" BC::go run ", run, " called\n");
    std::string timer_str("Timer_" + std::to_string(run));

    uint64_t loop_end = 1;
    bool sSources = false;

    if (!singleSourceBC) {
      if (!numberOfSources) {
        loop_end = graph.size();
      } else {
        loop_end = numberOfSources;
      }

      // if provided a file of sources to work with, use that
      if (sourceVector.size() != 0) {
        if (loop_end > sourceVector.size()) {
          loop_end = sourceVector.size();
        }
        sSources = true;
      }
    }

    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    for (uint64_t i = 0; i < loop_end; i++) {
      if (singleSourceBC) {
        // only 1 source; specified start source in command line
        assert(loop_end == 1);
        galois::gDebug("This is single source node BC");
        current_src_node = startSource;
      } else if (sSources) {
        current_src_node = sourceVector[i];
      } else {
        // all sources
        current_src_node = i;
      }

      globalRoundNumber = 0;
      backRoundCount = 0;

      StatTimer_main.start();
      BC::go(graph, dga);
      StatTimer_main.stop();

      // Round reporting
        galois::runtime::reportStat_Single(REGION_NAME,
          ("NumRounds"), globalRoundNumber);
        uint32_t backRounds;
        if (globalRoundNumber > 2) {
          backRounds = globalRoundNumber - 2;
        } else {
          backRounds = 0;
        }
        galois::runtime::reportStat_Single(REGION_NAME,
          ("NumForwardRounds"),
          globalRoundNumber);
        galois::runtime::reportStat_Single(REGION_NAME,
          ("NumBackRounds"), backRounds);
        galois::runtime::reportStat_Single(REGION_NAME,
          std::string("TotalRounds_") + std::to_string(run),
          globalRoundNumber + backRounds);
    }

    Sanity::go(graph, dga_max, dga_min, dga_sum);

    // re-init graph for next run
    if ((run + 1) != numRuns) {
      {
        bitset_num_shortest_paths.reset();
        bitset_current_length.reset();
        bitset_dependency.reset();
      }

      InitializeGraph::go((graph));
    }
  }

  StatTimer_total.stop();

  // Verify, i.e. print out graph data for examination
  if (verify) {
    char* v_out = (char*)malloc(40);
      for (auto ii = (graph).begin();
           ii != (graph).end(); ++ii) {
        // outputs betweenness centrality
        sprintf(v_out, "%lu %.9f\n", (*ii),
                (graph).getData(*ii).betweeness_centrality);

        galois::gPrint(v_out);
      }
    free(v_out);
  }

  return 0;
}
