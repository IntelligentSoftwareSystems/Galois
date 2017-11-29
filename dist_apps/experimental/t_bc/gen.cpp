/** Betweeness Centrality (Theoretical) -*- C++ -*-
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
 * @section Description
 *
 * Compute Betweeness-Centrality on distributed Galois; Vijaya Ramachandran's
 * theoretical BC 
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */


constexpr static const char* const REGION_NAME = "T_BC";

/******************************************************************************/
/* Sync code/calls was manually written, not compiler generated */
/******************************************************************************/

#include <iostream>
#include <limits>

#include "galois/DistGalois.h"
#include "galois/gstl.h"
#include "DistBenchStart.h"
#include "galois/DistAccumulator.h"
#include "galois/runtime/Tracer.h"

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/
namespace cll = llvm::cl;
static cll::opt<unsigned int> maxIterations("maxIterations", 
                               cll::desc("Maximum iterations: Default 10000"), 
                               cll::init(10000));
static cll::opt<bool> singleSourceBC("singleSource", 
                                cll::desc("Use for single source BC"),
                                cll::init(false));
static cll::opt<unsigned int> startSource("srcNodeId", 
                                cll::desc("Starting source node used for "
                                          "betweeness-centrality"),
                                cll::init(0));
static cll::opt<unsigned int> NUM_SOURCES("numOfSources", 
                                cll::desc("Number of sources to use for "
                                          "betweeness-centraility"),
                                cll::init(0));
static cll::opt<unsigned int> vIndex("index", 
                                cll::desc("Index to print min distance of"),
                                cll::init(0));

/******************************************************************************/
/* Graph structure declarations */
/******************************************************************************/

const uint32_t infinity = std::numeric_limits<uint32_t>::max() / 4;
static uint64_t current_src_node = 0;

// NOTE: types assume that these values will not reach uint64_t: it may
// need to be changed for very large graphs
struct NodeData {
  std::vector<uint32_t> oldMinDistances;

  //std::vector<uint32_t> minDistances;
  std::atomic<uint32_t>* minDistances;

  std::vector<uint32_t> shortestPathNumbers;
  std::vector<char> readyStatus;
  std::vector<uint32_t> sToSend;
  std::vector<uint32_t> savedRoundNumbers;

  std::atomic<float>* dependencyValues;
  //galois::LargeArray<std::atomic<float>> dependencyValues;

  //std::vector<float> dependencyValues;
  float bc;
};

static std::set<uint64_t> random_sources = std::set<uint64_t>();

typedef hGraph<NodeData, void, true> Graph;

typedef typename Graph::GraphNode GNode;

// bitsets for tracking updates
galois::DynamicBitSet bitset_to_add;
galois::DynamicBitSet bitset_to_add_float;
galois::DynamicBitSet bitset_num_shortest_paths;
galois::DynamicBitSet bitset_num_successors;
galois::DynamicBitSet bitset_num_predecessors;
galois::DynamicBitSet bitset_trim;
galois::DynamicBitSet bitset_current_length;
galois::DynamicBitSet bitset_propogation_flag;
galois::DynamicBitSet bitset_dependency;

// sync structures
#include "gen_sync.hh"

/******************************************************************************/
/* Functors for running the algorithm */
/******************************************************************************/
uint64_t offset;
uint32_t roundNumber = 0;
uint32_t totalSortTime = 0;

struct InitializeGraph {
  Graph *graph;

  InitializeGraph(Graph* _graph) : graph(_graph){}

  /* Initialize the graph */
  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();

    galois::do_all(
      // pass in begin/end to not use local thread ranges
      galois::iterate(allNodes.begin(), allNodes.end()), 
      InitializeGraph{&_graph}, 
      galois::loopname("InitializeGraph"), 
      //galois::loopname(_graph.get_run_identifier("InitializeGraph").c_str()), 
      galois::no_stats()
    );
  }

  /* Functor passed into the Galois operator to carry out initialization;
   * reset everything */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    src_data.oldMinDistances.resize(NUM_SOURCES);

    //src_data.minDistances.resize(NUM_SOURCES);

    src_data.minDistances = 
     (std::atomic<uint32_t>*)malloc(sizeof(std::atomic<uint32_t>) * NUM_SOURCES);

    src_data.shortestPathNumbers.resize(NUM_SOURCES);
    src_data.readyStatus.resize(NUM_SOURCES);
    src_data.sToSend.resize(NUM_SOURCES);
    src_data.savedRoundNumbers.resize(NUM_SOURCES);
    src_data.dependencyValues = 
     (std::atomic<float>*)malloc(sizeof(std::atomic<float>) * NUM_SOURCES);

    assert(src_data.dependencyValues != 0);

    src_data.bc = 0.0;
  }
};

/* This is used to reset node data when switching to a different NUM_SOURCES source set */
struct InitializeIteration {
  const uint32_t &local_infinity;
  Graph *graph;

  InitializeIteration(const uint32_t &_local_infinity,
                      const uint64_t &_local_current_src_node,
                      Graph* _graph) 
    : local_infinity(_local_infinity), graph(_graph) { }

  /* Reset necessary graph metadata */
  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();

    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()), 
      InitializeIteration{infinity, current_src_node, &_graph},
      galois::loopname("InitializeIteration"), 
      galois::no_stats()
    );
  }

  /* Functor passed into the Galois operator to carry out reset of node data
   * (aside from betweeness centrality measure) */
  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    for (unsigned i = 0; i < NUM_SOURCES; i++) {
      if ((offset + i) == graph->getGID(src)) {
        src_data.minDistances[i] = 0;
        src_data.shortestPathNumbers[i] = 1;
      } else {
        src_data.minDistances[i] = local_infinity;
        src_data.shortestPathNumbers[i] = 0;
      }
      src_data.readyStatus[i] = 1;
      src_data.sToSend[i] = 0;
      src_data.savedRoundNumbers[i] = local_infinity;
      src_data.dependencyValues[i] = 0.0;
      src_data.oldMinDistances[i] = src_data.minDistances[i];
    }

    // old min distance setting
    //for (unsigned i = 0; i < NUM_SOURCES; i++) {
    //  src_data.oldMinDistances[i] = src_data.minDistances[i];
    //}
    //src_data.oldMinDistances = src_data.minDistances;
  }
};

struct MidUpdate {
  Graph* graph;
  MidUpdate(Graph* _graph) : graph(_graph) { }

  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();
    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()), 
      MidUpdate(&_graph),
      galois::loopname("MidUpdate"),
      galois::no_stats()
    );
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    for (unsigned i = 0; i < NUM_SOURCES; i++) {
      if (src_data.oldMinDistances[i] != src_data.minDistances[i]) {
        src_data.shortestPathNumbers[i] = 0;
        src_data.readyStatus[i] = 1;
      }
    }
  }
};

struct APSP2 {
  Graph* graph;
  APSP2(Graph* _graph) : graph(_graph) { }

  void static go(Graph& _graph) {
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();
    galois::do_all(
      galois::iterate(nodesWithEdges), 
      APSP2(&_graph),
      galois::loopname("APSP2"),
      galois::no_stats()
    );
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    for (auto outEdge : graph->edges(src)) {
      GNode dst = graph->getEdgeDst(outEdge);
      auto& dnode = graph->getData(dst);
      // at this point minDist should be synchronized across all hosts
      for (unsigned i = 0; i < NUM_SOURCES; i++) {
        if (src_data.sToSend[i] != 0) {
          if ((src_data.oldMinDistances[i] + 1) == dnode.minDistances[i]) {
            dnode.shortestPathNumbers[i] += src_data.sToSend[i];
          }
        }
      }
    }
  }
};

struct Swap {
  Graph* graph;
  Swap(Graph* _graph) : graph(_graph) { }

  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();
    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()), 
      Swap(&_graph),
      galois::loopname("Swap"),
      galois::no_stats()
    );
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    for (unsigned i = 0; i < NUM_SOURCES; i++) {
      src_data.oldMinDistances[i] = src_data.minDistances[i];
    }
    //src_data.oldMinDistances = src_data.minDistances;
  }
};

struct APSP {
  const uint32_t local_infinity;
  uint32_t& roundNum;
  Graph* graph;
  galois::DGAccumulator<uint32_t>& DGAccumulator_accum;
  galois::StatTimer& t;

  APSP(const uint32_t _local_infinity, uint32_t& _roundNum, Graph* _graph, 
       galois::DGAccumulator<uint32_t>& dga, galois::StatTimer& _t) 
    : local_infinity(_local_infinity), roundNum(_roundNum), graph(_graph), 
      DGAccumulator_accum(dga), t(_t) { }

  void static go(Graph& _graph, galois::DGAccumulator<uint32_t>& dga,
                 galois::StatTimer& sortTimer) {
    const auto& nodesWithEdges = _graph.allNodesWithEdgesRange();

    do {
      dga.reset();
      //galois::gPrint("Round ", roundNumber, "\n");
      galois::do_all(
        galois::iterate(nodesWithEdges), 
        APSP(infinity, roundNumber, &_graph, dga, sortTimer),
        galois::loopname("APSP"),
        galois::no_stats()
      );

      // graph sync here
      MidUpdate::go(_graph);
      APSP2::go(_graph);
      // graph sync here
      Swap::go(_graph);

      roundNumber++;
    } while (dga.reduce());
  }

  struct DWrapper {
    uint32_t dist;
    uint32_t index;
    
    DWrapper(uint32_t _dist, uint32_t _index)
      : dist(_dist), index(_index) { }

    bool operator<(const DWrapper& b) const {
      return dist < b.dist;
    }
  };

  std::vector<DWrapper> 
  wrapDistVector(const std::vector<uint32_t>& dVector) const {
    std::vector<DWrapper> wrappedVector;
    wrappedVector.reserve(NUM_SOURCES);

    for (unsigned i = 0; i < NUM_SOURCES; i++) {
      wrappedVector.emplace_back(DWrapper(dVector[i], i));
    }
    assert(dVector.size() == wrappedVector.size());

    return wrappedVector;
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    std::vector<DWrapper> toSort = wrapDistVector(src_data.oldMinDistances);
    t.start();
    std::stable_sort(toSort.begin(), toSort.end());
    t.stop();

    totalSortTime += t.get();

    bool readyFound = false;
    uint32_t indexToSend = NUM_SOURCES + 1;

    for (unsigned i = 0; i < NUM_SOURCES; i++) {
      DWrapper& currentSource = toSort[i];
      uint32_t currentIndex = currentSource.index;

      // see if ready; use it if so
      if (!readyFound) {
        if (src_data.readyStatus[currentIndex]) {
          indexToSend = currentIndex;
          readyFound = true;
        }
      }

      // determine if we need to send out the shortest path length in this round
      if (i + currentSource.dist == roundNum) {
        // save round num
        src_data.savedRoundNumbers[currentIndex] = roundNum;

        // note that this index needs to have shortest path number sent out
        // by saving it to another vector
        src_data.sToSend[currentIndex] = 
            src_data.shortestPathNumbers[currentIndex];
      } else {
        src_data.sToSend[currentIndex] = 0;
      }
    }

    // there is something to send to other nodes this round
    if (readyFound) {
      uint32_t distValue = src_data.oldMinDistances[indexToSend];
      uint32_t newValue = distValue + 1;

      for (auto outEdge : graph->edges(src)) {
        GNode dst = graph->getEdgeDst(outEdge);
        auto& dnode = graph->getData(dst);

        uint32_t oldValue = galois::atomicMin(dnode.minDistances[indexToSend],
                                              newValue);

        if (oldValue > newValue) {
          // TODO set bitset
        }
      }
      DGAccumulator_accum += 1;
      src_data.readyStatus[indexToSend] = 0;
    }

  }
};

struct RoundUpdate {
  Graph* graph;
  const uint32_t roundNum;
  const uint32_t local_infinity;
  RoundUpdate(Graph* _graph, const uint32_t _roundNum, const uint32_t _li) 
    : graph(_graph), roundNum(_roundNum), local_infinity(_li) { }

  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();
    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()), 
      RoundUpdate(&_graph, roundNumber, infinity),
      galois::loopname("RoundUpdate"),
      galois::no_stats()
    );
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    for (unsigned i = 0; i < NUM_SOURCES; i++) {
      if (src_data.oldMinDistances[i] < local_infinity) {
        src_data.savedRoundNumbers[i] = 
            roundNum - src_data.savedRoundNumbers[i];
        assert(src_data.savedRoundNumbers[i] <= roundNum);
      }
    }
  }
};


struct BackProp {
  Graph* graph;
  const uint32_t roundNum;

  BackProp(Graph* _graph, const uint32_t _roundNum)
    : graph(_graph), roundNum(_roundNum) { }

  void static go(Graph& _graph) {
    const auto& allNodesWithEdgesIn = _graph.allNodesWithEdgesRangeIn();

    uint32_t currentRound = 0;

    while (currentRound <= roundNumber) {
      galois::do_all(
        galois::iterate(allNodesWithEdgesIn),
        BackProp(&_graph, currentRound),
        galois::loopname("BackProp"),
        galois::no_stats()
      );

      currentRound++;
    }
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    std::vector<uint32_t> toBackProp;

    for (unsigned i = 0; i < NUM_SOURCES; i++) {
      if (src_data.savedRoundNumbers[i] == roundNum) {
        toBackProp.emplace_back(i);
      }
    }

    // TODO why is this assertion failing?
    //assert(toBackProp.size() == 1);

    for (auto i : toBackProp) {
      uint32_t myDistance = src_data.oldMinDistances[i];

      // calculate final dependency value
      src_data.dependencyValues[i] = src_data.dependencyValues[i] * 
                                     src_data.shortestPathNumbers[i];

      // get the value to add to predecessors
      float toAdd = ((float)1 + src_data.dependencyValues[i]) / 
                      src_data.shortestPathNumbers[i];

      for (auto inEdge : graph->in_edges(src)) {
        GNode dst = graph->getInEdgeDst(inEdge);
        auto& dnode = graph->getData(dst);

        // determine if this dnode is a predecessor
        if (myDistance == (dnode.oldMinDistances[i] + 1)) {
          galois::atomicAdd(dnode.dependencyValues[i], toAdd);
        }
      }
    }
  }
};

struct BC {
  Graph* graph;

  BC(Graph* _graph) : graph(_graph) { }

  void static go(Graph& _graph) {
    const auto& allNodes = _graph.allNodesRange();

    galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()), 
      BC(&_graph),
      galois::loopname("BC"),
      galois::no_stats()
    );
  }

  void operator()(GNode src) const {
    NodeData& src_data = graph->getData(src);

    for (unsigned i = 0; i < NUM_SOURCES; i++) {
      // exclude sources themselves 
      if (graph->getGID(src) != (i + offset)) {
        src_data.bc += src_data.dependencyValues[i];
      }
    }
  }
};

/******************************************************************************/
/* Sanity check */
/******************************************************************************/

//struct Sanity {
//  Graph* graph;
//
//  static float current_max;
//  static float current_min;
//
//  galois::DGAccumulator<float>& DGAccumulator_max;
//  galois::DGAccumulator<float>& DGAccumulator_min;
//  galois::DGAccumulator<double>& DGAccumulator_sum;
//
//  Sanity(Graph* _graph,
//      galois::DGAccumulator<float>& _DGAccumulator_max,
//      galois::DGAccumulator<float>& _DGAccumulator_min,
//      galois::DGAccumulator<double>& _DGAccumulator_sum
//  ) : 
//    graph(_graph),
//    DGAccumulator_max(_DGAccumulator_max),
//    DGAccumulator_min(_DGAccumulator_min),
//    DGAccumulator_sum(_DGAccumulator_sum) {}
//
//  void static go(Graph& _graph,
//    galois::DGAccumulator<float>& DGA_max,
//    galois::DGAccumulator<float>& DGA_min,
//    galois::DGAccumulator<double>& DGA_sum
//  ) {
//  #ifdef __GALOIS_HET_CUDA__
//    if (personality == GPU_CUDA) {
//      // TODO currently no GPU support for sanity check operator
//      fprintf(stderr, "Warning: No GPU support for sanity check; might get "
//                      "wrong results.\n");
//    }
//  #endif
//
//    DGA_max.reset();
//    DGA_min.reset();
//    DGA_sum.reset();
//
//    galois::do_all(galois::iterate(_graph.allNodesRange().begin(), 
//                                   _graph.allNodesRange().end()),
//                   Sanity(
//                     &_graph,
//                     DGA_max,
//                     DGA_min,
//                     DGA_sum
//                   ), 
//                   galois::loopname("Sanity"),
//                   galois::no_stats());
//
//    DGA_max = current_max;
//    DGA_min = current_min;
//
//    float max_bc = DGA_max.reduce_max();
//    float min_bc = DGA_min.reduce_min();
//    double bc_sum = DGA_sum.reduce();
//
//    // Only node 0 will print data
//    if (galois::runtime::getSystemNetworkInterface().ID == 0) {
//      printf("Max BC is %f\n", max_bc);
//      printf("Min BC is %f\n", min_bc);
//      printf("BC sum is %f\n", bc_sum);
//    }
//  }
//  
//  /* Gets the max, min rank from all owned nodes and
//   * also the sum of ranks */
//  void operator()(GNode src) const {
//    NodeData& sdata = graph->getData(src);
//
//    if (graph->isOwned(graph->getGID(src))) {
//      if (current_max < sdata.betweeness_centrality) {
//        current_max = sdata.betweeness_centrality;
//      }
//
//      if (current_min > sdata.betweeness_centrality) {
//        current_min = sdata.betweeness_centrality;
//      }
//
//      DGAccumulator_sum += sdata.betweeness_centrality;
//    }
//  }
//};
//float Sanity::current_max = 0;
//float Sanity::current_min = std::numeric_limits<float>::max() / 4;

/******************************************************************************/
/* Main method for running */
/******************************************************************************/

constexpr static const char* const name = "Betweeness Centrality"; 
constexpr static const char* const desc = "Betweeness Centrality on Distributed "
                                          "Galois.";
constexpr static const char* const url = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  auto& net = galois::runtime::getSystemNetworkInterface();
  if (net.ID == 0) {
    galois::runtime::reportParam(REGION_NAME, "Max Iterations", 
                                (unsigned long)maxIterations);
  }

  galois::StatTimer StatTimer_total("TIMER_TOTAL", REGION_NAME);

  StatTimer_total.start();

  #ifdef __GALOIS_HET_CUDA__
  Graph* hg = twoWayDistGraphInitialization<NodeData, void>(&cuda_ctx);
  #else
  Graph* hg = twoWayDistGraphInitialization<NodeData, void>();
  #endif

  //bitset_to_add.resize(h_graph->size());
  //bitset_to_add_float.resize(h_graph->size());
  //bitset_num_shortest_paths.resize(h_graph->size());
  //bitset_trim.resize(h_graph->size());
  //bitset_dependency.resize(h_graph->size());

  galois::gPrint("[", net.ID, "] InitializeGraph::go called\n");

  galois::StatTimer StatTimer_graph_init("TIMER_GRAPH_INIT", REGION_NAME);

  StatTimer_graph_init.start();
  InitializeGraph::go(*hg);
  StatTimer_graph_init.stop();

  galois::runtime::getHostBarrier().wait();

  // shared DG accumulator among all steps
  galois::DGAccumulator<uint32_t> dga;

  // sanity dg accumulators
  //galois::DGAccumulator<float> dga_max;
  //galois::DGAccumulator<float> dga_min;
  //galois::DGAccumulator<double> dga_sum;

  offset = 0;

  galois::StatTimer sortTimer("SortTimer");
  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] Run ", run, " started\n");
    std::string timer_str("TIMER_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    StatTimer_main.start();

    InitializeIteration::go(*hg);
    APSP::go(*hg, dga, sortTimer);
    roundNumber--; // terminating round; i.e. last round
    RoundUpdate::go(*hg);
    BackProp::go(*hg);
    BC::go(*hg);

    StatTimer_main.stop();

    //Sanity::current_max = 0;
    //Sanity::current_min = std::numeric_limits<float>::max() / 4;

    //Sanity::go(
    //  *h_graph,
    //  dga_max,
    //  dga_min,
    //  dga_sum
    //);

    // re-init graph for next run
    if ((run + 1) != numRuns) {
      galois::runtime::getHostBarrier().wait();
      (*hg).set_num_run(run + 1);
      offset = 0;
      roundNumber = 0;

      //bitset_to_add.reset();
      //bitset_to_add_float.reset();
      //bitset_num_shortest_paths.reset();
      //bitset_num_successors.reset();
      //bitset_num_predecessors.reset();
      //bitset_trim.reset();
      //bitset_current_length.reset();
      //bitset_propogation_flag.reset();
      //bitset_dependency.reset();

      InitializeGraph::go(*hg);
      galois::runtime::getHostBarrier().wait();
    }
  }

  StatTimer_total.stop();

  //galois::gPrint("total sort time ", sortTimer.get(), "\n");

  // Verify, i.e. print out graph data for examination
  if (verify) {
    char *v_out = (char*)malloc(40);
    for (auto ii = (*hg).masterNodesRange().begin(); 
              ii != (*hg).masterNodesRange().end(); 
              ++ii) {
      //sprintf(v_out, "%lu %u\n", (*hg).getGID(*ii),
      //        (*hg).getData(*ii).minDistances[vIndex]);
      // outputs betweenness centrality
      sprintf(v_out, "%lu %.9f\n", (*hg).getGID(*ii),
              (*hg).getData(*ii).bc);
      //sprintf(v_out, "%lu %u\n", (*hg).getGID(*ii),
      //        (*hg).getData(*ii).shortestPathNumbers[0]);

      galois::runtime::printOutput(v_out);
    }
    free(v_out);
  }

  return 0;
}
