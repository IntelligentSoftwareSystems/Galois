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

constexpr static const char* const REGION_NAME = "MRBC";

//#define ENABLE_PAGE_REPORT
//#define USE_PREALLOC
//Go to line:230 to configure preAlloc

#include "galois/DistGalois.h"
#include "galois/DReducible.h"
#include "galois/runtime/Tracer.h"
#include "DistBenchStart.h"

#include <iostream>

// type of short path
using ShortPathType = double;

/**
 * Structure for holding data calculated during BC
 */
struct BCData {
  uint32_t minDistance;
  ShortPathType shortPathCount;
  galois::CopyableAtomic<float> dependencyValue;
};

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/
namespace cll = llvm::cl;

static cll::opt<std::string> sourcesToUse("sourcesToUse",
                                cll::desc("Sources to use in BC"),
                                cll::init(""));
static cll::opt<unsigned int> numSourcesPerRound("numRoundSources",
                                cll::desc("Number of sources to use for APSP"),
                                cll::init(1));
static cll::opt<unsigned int> totalNumSources("numOfSources",
                                cll::desc("Total number of sources to do BC"),
                                cll::init(0));
static cll::opt<bool> useSingleSource("singleSource",
                                cll::desc("Use a single source."),
                                cll::init(false));
static cll::opt<unsigned long long> startNode("startNode",
                                cll::desc("Single source start node."),
                                cll::init(0));
static cll::opt<unsigned int> vIndex("index",
                                cll::desc("DEBUG: Index to print for "
                                          "dist/short paths"),
                                cll::init(0), cll::Hidden);
// debug vars
static cll::opt<bool> outputDistPaths("outputDistPaths",
                                cll::desc("DEBUG: Output min distance"
                                          "/short path counts instead"),
                                cll::init(false), cll::Hidden);
static cll::opt<unsigned int> vectorSize("vectorSize",
                                cll::desc("DEBUG: Specify size of vector "
                                          "used for node data"),
                                cll::init(0), cll::Hidden);

// moved here so PRBCTree has access to numSourcesPerRound
#include "PRBCTree.h"

/******************************************************************************/
/* Graph structure declarations */
/******************************************************************************/

// NOTE: declared types assume that these values will not reach uint64_t: it may
// need to be changed for very large graphs
struct NodeData {
  galois::gstl::Vector<BCData> sourceData;
  // distance map
  PRBCTree dTree;
  // final bc value
  float bc;
  // index that needs to be pulled in a round
  uint32_t roundIndexToSend;
};

using Graph = galois::graphs::DistGraph<NodeData, void>;
using GNode = typename Graph::GraphNode;

// Bitsets for tracking which nodes need to be sync'd with respect to a
// particular field
galois::DynamicBitSet bitset_minDistances;
galois::DynamicBitSet bitset_dependency;

// moved here for access to ShortPathType, NodeData, DynamicBitSets
#include "pr_bc_opt_sync.hh"

/******************************************************************************/
/* Functions for running the algorithm */
/******************************************************************************/

/**
 * Graph initialization. Initialize all of the node data fields.
 *
 * @param graph Local graph to operate on
 */
void InitializeGraph(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();

  galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      [&](GNode curNode) {
        NodeData& cur_data = graph.getData(curNode);
        cur_data.sourceData.resize(vectorSize);
        cur_data.bc = 0.0;
      },
      galois::loopname(graph.get_run_identifier("InitializeGraph").c_str()),
      galois::no_stats()); // Only stats the runtime by loopname
}

/**
 * This is used to reset node data when switching to a different
 * source set. Initializes everything for the coming source set.
 *
 * @param graph Local graph to operate on
 * @param offset Offset into sources (i.e. number of sources already done)
 **/
void InitializeIteration(Graph& graph,
                         const std::vector<uint64_t>& nodesToConsider) {
  const auto& allNodes = graph.allNodesRange();

  galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      [&](GNode curNode) {
        NodeData& cur_data = graph.getData(curNode);
        cur_data.roundIndexToSend = infinity;
        cur_data.dTree.initialize();
        for (unsigned i = 0; i < numSourcesPerRound; i++) {
          // min distance and short path count setup
          if (nodesToConsider[i] == graph.getGID(curNode)) { // source node
            cur_data.sourceData[i].minDistance = 0;
            cur_data.sourceData[i].shortPathCount = 1;
            cur_data.sourceData[i].dependencyValue = 0.0;
            cur_data.dTree.setDistance(i, 0);
          } else { // non-source node
            cur_data.sourceData[i].minDistance = infinity;
            cur_data.sourceData[i].shortPathCount = 0;
            cur_data.sourceData[i].dependencyValue = 0.0;
          }
        }
      },
      galois::loopname(graph.get_run_identifier("InitializeIteration").c_str()),
      galois::no_stats());
};

/**
 * Find the message to send out from each node every round (if any exists to
 * be sent).
 *
 * @param graph Local graph to operate on
 * @param roundNumber current round number
 * @param dga Distributed accumulator for determining if work was done in
 * an iteration across all hosts
 */
void FindMessageToSync(Graph& graph, const uint32_t roundNumber,
                       galois::DGAccumulator<uint32_t>& dga) {
  const auto& allNodes = graph.allNodesRange();

  galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      [&](GNode curNode) {
        NodeData& cur_data = graph.getData(curNode);
        cur_data.roundIndexToSend = cur_data.dTree.getIndexToSend(roundNumber);

        if (cur_data.roundIndexToSend != infinity) {
          if (cur_data.sourceData[cur_data.roundIndexToSend].minDistance != 0) {
            bitset_minDistances.set(curNode);
          }
          dga += 1;
        } else if (cur_data.dTree.moreWork()) {
          dga += 1;
        }
      },
      galois::loopname(
          graph.get_run_identifier("FindMessageToSync").c_str()),
      galois::steal(),
      galois::no_stats());
}

/**
 * Mark index we're sending out this round as sent + update metadata as necessary.
 *
 * @param graph Local graph to operate on
 * @param roundNumber current round number
 * @param dga Distributed accumulator for determining if work was done in
 * an iteration across all hosts
 */
void ConfirmMessageToSend(Graph& graph, const uint32_t roundNumber,
                          galois::DGAccumulator<uint32_t>& dga) {
  const auto& allNodes = graph.allNodesRange();

  galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      [&](GNode curNode) {
        NodeData& cur_data = graph.getData(curNode);
        if (cur_data.roundIndexToSend != infinity) {
          cur_data.dTree.markSent(roundNumber);
        }
      },
      galois::loopname(
          graph.get_run_identifier("ConfirmMessageToSend").c_str()),
      galois::no_stats());
}

/**
 * If a node has something to send (as indicated by its indexToSend variable),
 * it will be pulled by all of its outgoing neighbors.
 *
 * Pull-style is used here to avoid the need for locks as 2 variables must be
 * updated at once.
 *
 * @param graph Local graph to operate on
 * @param dga Distributed accumulator for determining if work was done in
 * an iteration across all hosts
 */
void SendAPSPMessagesOp(GNode dst, Graph& graph, galois::DGAccumulator<uint32_t>& dga) {
  auto& dnode = graph.getData(dst);
  auto& dnodeData = dnode.sourceData;

  for (auto inEdge : graph.edges(dst)) {
    NodeData& src_data   = graph.getData(graph.getEdgeDst(inEdge));
    uint32_t indexToSend = src_data.roundIndexToSend;

    if (indexToSend != infinity) {
      uint32_t distValue = src_data.sourceData[indexToSend].minDistance;
      uint32_t newValue  = distValue + 1;
      // Update minDistance vector
      auto& dnodeIndex = dnodeData[indexToSend];
      uint32_t oldValue = dnodeIndex.minDistance;

      if (oldValue > newValue) {
        dnodeIndex.minDistance = newValue;
        dnode.dTree.setDistance(indexToSend, oldValue, newValue);
        // overwrite short path with this node's shortest path
        dnodeIndex.shortPathCount =
            src_data.sourceData[indexToSend].shortPathCount;
      } else if (oldValue == newValue) {
        assert(src_data.sourceData[indexToSend].shortPathCount != 0);
        // add to short path
        dnodeIndex.shortPathCount +=
            src_data.sourceData[indexToSend].shortPathCount;
      }

      dga += 1;
    }
  }
}

void SendAPSPMessages(Graph& graph, galois::DGAccumulator<uint32_t>& dga) {
  const auto& allNodesWithEdges = graph.allNodesWithEdgesRange();

  galois::do_all(
      galois::iterate(allNodesWithEdges),
      [&](GNode dst) {
        SendAPSPMessagesOp(dst, graph, dga);
      },
      galois::loopname(
          graph.get_run_identifier("SendAPSPMessages").c_str()),
      galois::steal(),
      galois::no_stats());
}

/**
 * Find all pairs shortest paths for the sources currently being worked on
 * as well as the number of shortest paths for each source.
 *
 * @param graph Local graph to operate on
 * @param dga Distributed accumulator for determining if work was done in
 * an iteration across all hosts
 *
 * @returns total number of rounds needed to do this phase
 */
uint32_t APSP(Graph& graph, galois::DGAccumulator<uint32_t>& dga) {
  uint32_t roundNumber = 0;

  do {
    dga.reset();
    galois::gDebug("[", galois::runtime::getSystemNetworkInterface().ID, "]",
                   " Round ", roundNumber);
    graph.set_num_round(roundNumber);

    // you can think of this FindMessageToSync call being a part of the sync
    FindMessageToSync(graph, roundNumber, dga);

    // Template para's are struct names
    graph.sync<writeAny, readAny, APSPReduce, APSPBroadcast,
               Bitset_minDistances>(std::string("APSP"));

    // confirm message to send after sync potentially changes what you were
    // planning on sending
    ConfirmMessageToSend(graph, roundNumber, dga);

    // send messages (if any)
    SendAPSPMessages(graph, dga);

    roundNumber++;
  } while (dga.reduce(graph.get_run_identifier()));

  return roundNumber;
}

/**
 * Get the round number for the backward propagation phase using the round
 * number from the APSP phase. This round number determines when a node should
 * send out a message for the backward propagation of dependency values.
 *
 * @param graph Local graph to operate on
 */
void RoundUpdate(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();
  graph.set_num_round(0);

  galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      [&](GNode node) {
        NodeData& cur_data = graph.getData(node);
        cur_data.dTree.prepForBackPhase();
      },
      galois::loopname(
          graph.get_run_identifier("RoundUpdate").c_str()),
      galois::no_stats());
}

/**
 * Find the message that needs to be back propagated this round by checking
 * round number.
 */
void BackFindMessageToSend(Graph& graph, const uint32_t roundNumber,
                           const uint32_t lastRoundNumber) {
  // has to be all nodes because even nodes without edges may have dependency
  // that needs to be sync'd
  const auto& allNodes = graph.allNodesRange();

  galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      [&](GNode dst) {
        NodeData& dst_data        = graph.getData(dst);

        // if zero distances already reached, there is no point sending things
        // out since we don't care about dependecy for sources (i.e. distance
        // 0)
        if (!dst_data.dTree.isZeroReached()) {
          dst_data.roundIndexToSend =
            dst_data.dTree.backGetIndexToSend(roundNumber, lastRoundNumber);

          if (dst_data.roundIndexToSend != infinity) {
            // only comm if not redundant 0
            if (dst_data.sourceData[dst_data.roundIndexToSend].dependencyValue != 0) {
              bitset_dependency.set(dst);
            }
          }
        }
      },
      galois::loopname(
        graph.get_run_identifier("BackFindMessageToSend").c_str()
      ),
      galois::no_stats());
}

/**
 * Back propagate dependency values depending on the round that a node
 * sent out the shortest path message.
 *
 * @param graph Local graph to operate on
 * @param lastRoundNumber last round number in the APSP phase
 */
void BackPropOp(GNode dst, Graph& graph) {
  NodeData& dst_data = graph.getData(dst);
  unsigned i         = dst_data.roundIndexToSend;

  if (i != infinity) {
    uint32_t myDistance = dst_data.sourceData[i].minDistance;

    // calculate final dependency value
    dst_data.sourceData[i].dependencyValue =
      dst_data.sourceData[i].dependencyValue *
        dst_data.sourceData[i].shortPathCount;

    // get the value to add to predecessors
    float toAdd = ((float)1 + dst_data.sourceData[i].dependencyValue) /
                  dst_data.sourceData[i].shortPathCount;

    for (auto inEdge : graph.edges(dst)) {
      GNode src      = graph.getEdgeDst(inEdge);
      auto& src_data = graph.getData(src);

      uint32_t sourceDistance = src_data.sourceData[i].minDistance;

      // source nodes of this batch (i.e. distance 0) can be safely
      // ignored
      if (sourceDistance != 0) {
        // determine if this source is a predecessor
        if (myDistance == (sourceDistance + 1)) {
          // add to dependency of predecessor using our finalized one
          galois::atomicAdd(src_data.sourceData[i].dependencyValue, toAdd);
        }
      }
    }
  }
}
void BackProp(Graph& graph, const uint32_t lastRoundNumber) {
  // All nodes WITH EDGES (another at SendMessage)
  const auto& allNodesWithEdges = graph.allNodesWithEdgesRange();

  uint32_t currentRound = 0;

  while (currentRound <= lastRoundNumber) {
    graph.set_num_round(currentRound);

    BackFindMessageToSend(graph, currentRound, lastRoundNumber);

    // write destination in this case being the source in the actual graph
    // since we're using the tranpose graph
    graph.sync<writeDestination, readSource, DependencyReduce,
               DependencyBroadcast, Bitset_dependency>(
        std::string("DependencySync"));

    galois::do_all(
        galois::iterate(allNodesWithEdges),
        [&](GNode dst) {
          BackPropOp(dst, graph);
        },
        galois::loopname(
            graph.get_run_identifier("BackProp").c_str()),
        galois::steal(),
        galois::no_stats());

    currentRound++;
  }
}

/**
 * BC sum: take the dependency value for each source and add it to the
 * final BC value.
 *
 * @param graph Local graph to operate on
 * @param offset Offset into sources (i.e. number of sources already done)
 */
void BC(Graph& graph, const std::vector<uint64_t>& nodesToConsider) {
  const auto& masterNodes = graph.masterNodesRange();
  graph.set_num_round(0);

  galois::do_all(
      galois::iterate(masterNodes.begin(), masterNodes.end()),
      [&](GNode node) {
        NodeData& cur_data = graph.getData(node);

        for (unsigned i = 0; i < numSourcesPerRound; i++) {
          // exclude sources themselves from BC calculation
          if (graph.getGID(node) != nodesToConsider[i]) {
            cur_data.bc += cur_data.sourceData[i].dependencyValue;
          }
        }
      },
      galois::loopname(graph.get_run_identifier("BC").c_str()),
      galois::no_stats());
};

/******************************************************************************/
/* Sanity check */
/******************************************************************************/

void Sanity(Graph& graph) {
  galois::DGReduceMax<float> DGA_max;
  galois::DGReduceMin<float> DGA_min;
  galois::DGAccumulator<float> DGA_sum;

  DGA_max.reset();
  DGA_min.reset();
  DGA_sum.reset();

  galois::do_all(galois::iterate(graph.masterNodesRange().begin(),
                                 graph.masterNodesRange().end()),
                 [&](auto src) {
                   NodeData& sdata = graph.getData(src);

                   DGA_max.update(sdata.bc);
                   DGA_min.update(sdata.bc);
                   DGA_sum += sdata.bc;
                 },
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
};

/******************************************************************************/
/* Main method for running */
/******************************************************************************/

constexpr static const char* const name = "Min-Rounds Betweeness Centrality";
constexpr static const char* const desc = "Min-Rounds Betweeness "
                                          "Centrality on Distributed Galois.";
constexpr static const char* const url = 0;

uint64_t macroRound = 0; // macro round, i.e. number of batches done so far

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);
  auto& net = galois::runtime::getSystemNetworkInterface();

  #ifdef ENABLE_PAGE_REPORT
  galois::reportPageAlloc("SysStart");
  #endif

  // Total timer
  galois::StatTimer StatTimer_total("TimerTotal", REGION_NAME);
  StatTimer_total.start();

  galois::gPrint("[", net.ID, "] InitializeGraph\n");
  // false = iterate over in edges
  Graph* hg = distGraphInitialization<NodeData, void, false>();

  if (totalNumSources == 0) {
    galois::gDebug("Total num sources unspecified");
    totalNumSources = hg->globalSize();
  }

  if (useSingleSource) {
    totalNumSources    = 1;
    numSourcesPerRound = 1;
  }

  // set vector size in node data
  if (vectorSize == 0) {
    vectorSize = numSourcesPerRound;
  }
  GALOIS_ASSERT(vectorSize >= numSourcesPerRound);

  // Backup the number of sources per round
  uint64_t origNumRoundSources = numSourcesPerRound;

  // Start graph initialization
  galois::StatTimer StatTimer_graph_init("TIMER_GRAPH_INIT", REGION_NAME);
  StatTimer_graph_init.start();
  InitializeGraph(*hg);
  StatTimer_graph_init.stop();

  galois::runtime::getHostBarrier().wait();

  // shared DG accumulator among all steps
  galois::DGAccumulator<uint32_t> dga;

  // reading in list of sources to operate on if provided
  std::ifstream sourceFile;
  std::vector<uint64_t> sourceVector;
  if (sourcesToUse != "") {
    sourceFile.open(sourcesToUse);
    std::vector<uint64_t> t(std::istream_iterator<uint64_t>{sourceFile},
                            std::istream_iterator<uint64_t>{});
    sourceVector = t; // stored in source vector
    sourceFile.close();
  }

  // "sourceVector" if file not provided
  std::vector<uint64_t> nodesToConsider;
  nodesToConsider.resize(numSourcesPerRound);

  // bitset initialization
  bitset_dependency.resize(hg->size());
  bitset_minDistances.resize(hg->size());

////////////////////////////////////////////////////////////////////////////////

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] Run ", run, " started\n");

    /** Allocation - First time get here, seemingly graph-independent **
     ========= ========= ========= ========= ========= ========= =========
     Bat. Sz   1024      512       256       128       64        32
     ========= ========= ========= ========= ========= ========= =========
     Pg Alloc. 38195     19211     9635      4931      2634      1402
     ========= ========= ========= ========= ========= ========= =========
     Fitting: y = 37.0916 x + 205.224 (r2 = 0.999992)
     */
    #ifdef USE_PREALLOC
    /** Page Consumption - Indochina **
     ========= ========= ========= ========= ========= ========= =========
     Bat. Sz   1024      512       256       128       64        32
     ========= ========= ========= ========= ========= ========= =========
     Total Pg  59054     29718     15061     7722      4169      2426
     ========= ========= ========= ========= ========= ========= =========
     Fitting: y = 57.1598 x + 485.98 (r2 = 0.999989, sufficient bias: 596.89)
     */
    /** Page Consumption - LiveJournal **
     ========= ========= ========= ========= ========= ========= =========
     Bat. Sz   1024      512       256       128       64        32
     ========= ========= ========= ========= ========= ========= =========
     Total Pg  41239     21740     11959     7006      4460      3343
     ========= ========= ========= ========= ========= ========= =========
     Fitting: y = 38.2509 x + 2105.54 (r2 = 0.999984, sufficient bias: 2155.54)
     */
    // Related arg's: hg->size(), hg->sizeEdges(), numberThreads,
    // numSourcesPerRound
    // galois::runtime::pagePoolSize()
    // ghostwheel4 - HugePages_Total: 65536
    galois::StatTimer StatTimer_preAlloc("PreAlloc", REGION_NAME);
    StatTimer_preAlloc.start();
    galois::reportPageAlloc("MeminfoPre");
    // galois::preAlloc(20 * numSourcesPerRound + 392); // Indochina
    // galois::preAlloc(2 * numSourcesPerRound + 1951); // LiveJournal
    galois::reportPageAlloc("MeminfoPost");
    StatTimer_preAlloc.stop();
    #endif

    #ifdef ENABLE_PAGE_REPORT
    galois::reportPageAlloc(
      std::string("StartOfRun_" + std::to_string(run)).c_str());
    #endif

    // Timer per RUN
    std::string timer_str("Timer_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    // Associated to totalNumSources
    uint64_t totalSourcesFound = 0;

    // offset into sources to operate on
    uint64_t offset = 0;
    // node boundary to end search at
    uint64_t nodeBoundary = sourceVector.size() == 0 ?
                            hg->globalSize() :
                            sourceVector.size();

    while (offset < nodeBoundary && totalSourcesFound < totalNumSources) {
      if (useSingleSource) {
        nodesToConsider[0] = startNode;
      } else {
        unsigned sourcesFound = 0;
        while (sourcesFound < numSourcesPerRound &&
               offset < nodeBoundary &&
               totalSourcesFound < totalNumSources) {
          // choose from read source file or from beginning (0 to n)
          nodesToConsider[sourcesFound] = sourceVector.size() == 0 ?
                                          offset :
                                          sourceVector[offset];
          offset++;
          sourcesFound++;
          totalSourcesFound++;
        }

        if (sourcesFound == 0) {
          assert(offset == totalNumSources ||
                 totalSourcesFound == totalNumSources);
          break;
        }

        if (offset < totalNumSources) {
          assert(numSourcesPerRound == sourcesFound);
        } else {
          // >= totalNumSources
          assert(offset == totalNumSources);
          galois::gDebug("Out of sources (found ", sourcesFound, ")");
          numSourcesPerRound = sourcesFound;
        }
      }

      galois::gDebug("Using the following sources");
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        galois::gDebug(nodesToConsider[i]);
      }

      if (net.ID == 0) {
        galois::gPrint("Begin batch #", macroRound, "\n");
      }

      // accumulate time per batch
      StatTimer_main.start();
      InitializeIteration(*hg, nodesToConsider);

      // APSP returns total number of rounds taken
      // subtract 2 to get to last round where message was sent (round
      // after that is empty round where nothing is done)
      uint32_t lastRoundNumber = APSP(*hg, dga) - 2;
      RoundUpdate(*hg);
      BackProp(*hg, lastRoundNumber);
      BC(*hg, nodesToConsider);

      StatTimer_main.stop();

      hg->set_num_round(0);
      // report num rounds
      if (galois::runtime::getSystemNetworkInterface().ID == 0) {
        galois::runtime::reportStat_Single(REGION_NAME,
          //hg->get_run_identifier("NumForwardRounds", macroRound),
          hg->get_run_identifier("NumForwardRounds"),
          lastRoundNumber + 2);
        galois::runtime::reportStat_Single(REGION_NAME,
          //hg->get_run_identifier("NumBackwardRounds", macroRound),
          hg->get_run_identifier("NumBackwardRounds"),
          lastRoundNumber + 1);
        galois::runtime::reportStat_Single(REGION_NAME,
          hg->get_run_identifier("TotalRounds"),
          lastRoundNumber + lastRoundNumber + 3);
      }

      macroRound++;
    }

    Sanity(*hg);

    // re-init graph for next run
    if ((run + 1) != numRuns) { // not the last run
      galois::runtime::getHostBarrier().wait();
      (*hg).set_num_run(run + 1);
      (*hg).set_num_round(0);
      offset             = 0;
      macroRound = 0;
      numSourcesPerRound = origNumRoundSources;

      bitset_dependency.reset();
      bitset_minDistances.reset();

      InitializeGraph(*hg);
      galois::runtime::getHostBarrier().wait();
    }

    #if defined(USE_PREALLOC) || defined(ENABLE_PAGE_REPORT)
    galois::reportPageAlloc(
        std::string("EndOfRun_" + std::to_string(run)).c_str());
    #endif
    // Current run finished
  }

  StatTimer_total.stop();

////////////////////////////////////////////////////////////////////////////////

  // Verify, i.e. print out graph data for examination
  if (verify) {
    // buffer for text to be written out to file
    char* v_out = (char*)malloc(40);

    for (auto ii = (*hg).masterNodesRange().begin();
         ii != (*hg).masterNodesRange().end(); ++ii) {
      if (!outputDistPaths) {
        // outputs betweenness centrality
        sprintf(v_out, "%lu %.9f\n", (*hg).getGID(*ii), (*hg).getData(*ii).bc);
      } else {
        uint64_t a      = 0;
        ShortPathType b = 0;
        for (unsigned i = 0; i < numSourcesPerRound; i++) {
          if ((*hg).getData(*ii).sourceData[i].minDistance != infinity) {
            a += (*hg).getData(*ii).sourceData[i].minDistance;
          }
          b += (*hg).getData(*ii).sourceData[i].shortPathCount;
        }
      }

      galois::runtime::printOutput(v_out);
    }

    free(v_out);
  }

  #ifdef ENABLE_PAGE_REPORT
  galois::reportPageAlloc("SysTerm");
  #endif

  return 0;
}
