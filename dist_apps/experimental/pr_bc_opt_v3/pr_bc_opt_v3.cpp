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

#include "galois/DistGalois.h"
#include "galois/DReducible.h"
#include "galois/runtime/Tracer.h"
#include "DistBenchStart.h"

#include <iostream>

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

// type of short path
using ShortPathType = double;

/******************************************************************************/
/* Graph structure declarations */
/******************************************************************************/

// NOTE: declared types assume that these values will not reach uint64_t: it may
// need to be changed for very large graphs
struct NodeData {
  // current min distances for each source
  galois::gstl::Vector<uint32_t>                      minDistances;
  // actual shortest path number
  galois::gstl::Vector<ShortPathType>                 shortestPathNumbers;
  // dependency values
  galois::gstl::Vector<galois::CopyableAtomic<float>> dependencyValues;
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


constexpr static const char* const REGION_NAME = "PR_BC";
constexpr static const char* const name = "Pontecorvi-Ramachandran Betweeness "
                                          "Centrality";
constexpr static const char* const desc = "Pontecorvi-Ramachandran Betweeness "
                                          "Centrality on Distributed Galois.";
constexpr static const char* const url = 0;
uint64_t macroRound = 0; // macro round, i.e. number of batches done so far

/******************************************************************************/
/* Functions for running the algorithm */
/******************************************************************************/
void InitializeGraph(Graph& graph);
void InitializeIteration(Graph& graph, 
                        const std::vector<uint64_t>& nodesToConsider);
uint32_t APSP(Graph& graph, 
              galois::DGAccumulator<uint32_t>& dga);
void RoundUpdate(Graph& graph);
void BackProp(Graph& graph, 
              const uint32_t lastRoundNumber);
void BC(Graph& graph, 
        const std::vector<uint64_t>& nodesToConsider);
void Sanity(Graph& graph);

/******************************************************************************/
/* Main method for running */
/******************************************************************************/
int main(int argc, char** argv) {

  /* Initialize the distributed system */
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);
  auto& net = galois::runtime::getSystemNetworkInterface();
  // Total timer
  galois::StatTimer StatTimer_total("TimerTotal", REGION_NAME);
  StatTimer_total.start();
  /*************************************/

  /* Initialize Graph */
  galois::gPrint("[", net.ID, "] InitializeGraph\n");
  // false = iterate over in edges
  Graph* hg = distGraphInitialization<NodeData, void, false>();

  // Arg's config
  if (totalNumSources == 0) {
    galois::gDebug("Total num sources unspecified");
    totalNumSources = useSingleSource? 1 : hg->globalSize();
  }
  if (useSingleSource) {
    numSourcesPerRound = 1;
  }
  // set vector size in node data
  if (vectorSize == 0) { // minDistances, shortestPathNumbers, dependencyValues
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
  /****************************************************************************/

  galois::runtime::getHostBarrier().wait();

  /****************************************************************************/
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

/* RUN ************************************************************************/
  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] Run ", run, " started\n");
    // Timer per RUN
    std::string timer_str("Timer_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    // Associated to totalNumSources
    uint64_t totalSourcesFound = 0;

    // offset into sources to operate on
    uint64_t offset = 0;
    uint64_t numNodes = 
              sourceVector.size() == 0? hg->globalSize() : sourceVector.size();

    /* BATCH ****************************************************************/
    while (offset < numNodes && totalSourcesFound < totalNumSources) {
      // Prepare nodesToConsider
      if (useSingleSource) {
        nodesToConsider[0] = startNode;
      } else {
        unsigned sourcesFound = 0;
        while (sourcesFound < numSourcesPerRound // BATCH-level
                && offset < numNodes && totalSourcesFound < totalNumSources) {
          // determine sources to work on in this batch
          // no skip
          nodesToConsider[sourcesFound] = 
                        sourceVector.size() == 0? offset : sourceVector[offset];
          offset++;
          sourcesFound++; // in this BATCH
        }
        totalSourcesFound += sourcesFound; // in this RUN

        // correct numSourcesPerRound if not enough sources found
        if (offset == numNodes) {
          galois::gDebug("Out of sources (found ", sourcesFound, ")");
          numSourcesPerRound = sourcesFound;
        } else {
          // nodesToConsider full, there are still sources unfound
          assert(sourcesFound == numSourcesPerRound);
        }
      }

      /* DEBUG BLOCK *****************************************************/
      // Now nodesToConsider should be: 1. single source
      // 2. {offset, offset + 1, ..., sourcesPerRound - 1}
      // 3. copy from sourceFile
      galois::gDebug("Using the following sources");
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        galois::gDebug(nodesToConsider[i]);
      }
      /******************************************************************/

      // Start this BATCH
      if (net.ID == 0) {
        galois::gPrint("Begin batch #", macroRound, "\n");
      }

      // accumulate time per batch
      StatTimer_main.start();
      // BATCH start

      // 1
      InitializeIteration(*hg, nodesToConsider);

      // 2
      // APSP returns total number of rounds taken
      // subtract 2 to get to last round where message was sent
      // 1 for round after that is empty round where nothing is done
      // 1 for overheaded
      uint32_t lastRoundNumber = APSP(*hg, dga) - 2;

      // 3
      // Just for calling dTree.prepForBackProp()
      RoundUpdate(*hg);

      // 4
      BackProp(*hg, lastRoundNumber);

      // 5
      BC(*hg, nodesToConsider);

      // BATCH end
      StatTimer_main.stop();

      hg->set_num_round(0);
      // report num rounds
      if (galois::runtime::getSystemNetworkInterface().ID == 0) { // Pre-defined as net.ID /***/
        galois::runtime::reportStat_Single(REGION_NAME,
          //hg->get_run_identifier("NumForwardRounds", macroRound),
          hg->get_run_identifier("NumForwardRounds"),
          lastRoundNumber + 2);
        galois::runtime::reportStat_Single(REGION_NAME,
          //hg->get_run_identifier("NumBackwardRounds", macroRound),
          hg->get_run_identifier("NumBackwardRounds"),
          lastRoundNumber + 1);
        galois::runtime::reportStat_Tsum(REGION_NAME,
          hg->get_run_identifier("TotalRounds"),
          lastRoundNumber + lastRoundNumber + 3);
      }

      macroRound++;
    }

    // sanity
    Sanity(*hg);

    // re-init graph for next run
    if ((run + 1) != numRuns) { // not the last run
      galois::runtime::getHostBarrier().wait();

      (*hg).set_num_run(run + 1); // Manually set run number
      (*hg).set_num_round(0); // Manually set round number

      macroRound         = 0;

      numSourcesPerRound = origNumRoundSources;

      bitset_dependency.reset();
      bitset_minDistances.reset();

      InitializeGraph(*hg);
      galois::runtime::getHostBarrier().wait();
    }

    // Current run finished
  }

  StatTimer_total.stop();

  // All runs end
  /******************************************************************************/

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
        // sprintf(v_out, "%lu ", (*hg).getGID(*ii));
        // galois::runtime::printOutput(v_out);
        // for (unsigned i = 0; i < numSourcesPerRound; i++) {
        //  if ((*hg).getData(*ii).savedRoundNumbers[i] != infinity) {
        //    sprintf(v_out, "%u", (*hg).getData(*ii).savedRoundNumbers[i]);
        //    galois::runtime::printOutput(v_out);
        //  }
        //}
        ////sprintf(v_out, " ");
        ////galois::runtime::printOutput(v_out);

        ////for (unsigned i = 0; i < numSourcesPerRound; i++) {
        ////  sprintf(v_out, "%lu", (*hg).getData(*ii).shortestPathNumbers[i]);
        ////  galois::runtime::printOutput(v_out);
        ////}
        // sprintf(v_out, "\n");
        // galois::runtime::printOutput(v_out);

        uint64_t a      = 0;
        ShortPathType b = 0;
        for (unsigned i = 0; i < numSourcesPerRound; i++) {
          if ((*hg).getData(*ii).minDistances[i] != infinity) {
            a += (*hg).getData(*ii).minDistances[i];
          }
          b += (*hg).getData(*ii).shortestPathNumbers[i];
        }
        // outputs min distance and short path numbers
        // sprintf(v_out, "%lu %lu %lu %lu\n", (*hg).getGID(*ii), a, b, c);
        // sprintf(v_out, "%lu %lu %lu\n", (*hg).getGID(*ii), a, c);
        // sprintf(v_out, "%lu %lu\n", (*hg).getGID(*ii), b);
        // sprintf(v_out, "%lu %u %lu\n", (*hg).getGID(*ii),
        //                              (*hg).getData(*ii).minDistances[vIndex],
        //                              (*hg).getData(*ii).shortestPathNumbers[vIndex]);
      }

      galois::runtime::printOutput(v_out);
    }

    free(v_out);
  }

  return 0;
}

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

        cur_data.minDistances.resize(vectorSize);
        cur_data.shortestPathNumbers.resize(vectorSize);
        cur_data.dependencyValues.resize(vectorSize);
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
          cur_data.dependencyValues[i]  = 0.0;
          if (nodesToConsider[i] == graph.getGID(curNode)) { // source node
            cur_data.minDistances[i] = 0;
            // if (graph.isOwned(graph.L2G(curNode))) {
              cur_data.shortestPathNumbers[i] = 1; // Self-loop
            // } else {
            //   cur_data.shortestPathNumbers[i] = 0;
            // }
            cur_data.dTree.setDistance(i, 0); // Set dTree with entry (0, self)
          } else { // non-source node
            cur_data.minDistances[i]        = infinity;
            cur_data.shortestPathNumbers[i] = 0;
          }
        }
      },
      galois::loopname(graph.get_run_identifier("InitializeIteration").c_str()),
      galois::no_stats());
};

void FindMessageToSync(Graph& graph, 
                        const uint32_t roundNumber,
                        galois::DGAccumulator<uint32_t>& dga);
void ConfirmMessageToSend(Graph& graph, const uint32_t roundNumber,
                          galois::DGAccumulator<uint32_t>& dga);
void SendAPSPMessages(Graph& graph, galois::DGAccumulator<uint32_t>& dga);
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
          if (cur_data.minDistances[cur_data.roundIndexToSend] != 0) {
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
  // All nodes WITH EDGES (another at BackProp)
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
void SendAPSPMessages(Graph& graph, galois::DGAccumulator<uint32_t>& dga) {
  const auto& allNodesWithEdges = graph.allNodesWithEdgesRange();

  galois::do_all(
      galois::iterate(allNodesWithEdges),
      [&](GNode dst) {
        auto& dnode = graph.getData(dst);

        for (auto inEdge : graph.edges(dst)) {
          NodeData& src_data   = graph.getData(graph.getEdgeDst(inEdge));
          uint32_t indexToSend = src_data.roundIndexToSend;

          if (indexToSend != infinity) {
            uint32_t distValue = src_data.minDistances[indexToSend];
            uint32_t newValue  = distValue + 1;
            // Update minDistance vector
            uint32_t oldValue =
                galois::min(dnode.minDistances[indexToSend], newValue);

            // Update dTree & shortestPathNumbers
            if (oldValue > newValue) {
              dnode.dTree.setDistance(indexToSend, oldValue, newValue);
              // overwrite short path with this node's shortest path
              dnode.shortestPathNumbers[indexToSend] =
                  src_data.shortestPathNumbers[indexToSend];
            } else if (oldValue == newValue) {
              assert(src_data.shortestPathNumbers[indexToSend] != 0);
              // add to short path
              dnode.shortestPathNumbers[indexToSend] +=
                  src_data.shortestPathNumbers[indexToSend];
            }

            dga += 1;
          }
        }
      },
      galois::loopname(
          graph.get_run_identifier("SendAPSPMessages").c_str()),
      galois::steal(),
      galois::no_stats());
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

void BackFindMessageToSend(Graph& graph, const uint32_t roundNumber,
                           const uint32_t lastRoundNumber);
/**
 * Back propagate dependency values depending on the round that a node
 * sent out the shortest path message.
 *
 * @param graph Local graph to operate on
 * @param lastRoundNumber last round number in the APSP phase
 */
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
          NodeData& dst_data = graph.getData(dst);
          unsigned i         = dst_data.roundIndexToSend;

          if (i != infinity) {
            uint32_t myDistance = dst_data.minDistances[i];

            // calculate final dependency value
            dst_data.dependencyValues[i] =
                dst_data.dependencyValues[i] * dst_data.shortestPathNumbers[i];

            // get the value to add to predecessors
            float toAdd = ((float)1 + dst_data.dependencyValues[i]) /
                          dst_data.shortestPathNumbers[i];

            for (auto inEdge : graph.edges(dst)) {
              GNode src      = graph.getEdgeDst(inEdge);
              auto& src_data = graph.getData(src);

              uint32_t sourceDistance = src_data.minDistances[i];

              // source nodes of this batch (i.e. distance 0) can be safely 
              // ignored
              if (sourceDistance != 0) {
                // determine if this source is a predecessor
                if (myDistance == (sourceDistance + 1)) {
                  // add to dependency of predecessor using our finalized one
                  galois::atomicAdd(src_data.dependencyValues[i], toAdd);
                }
              }
            }
          }
        },
        galois::loopname(
            graph.get_run_identifier("BackProp").c_str()),
        galois::steal(),
        galois::no_stats());

    currentRound++;
  }
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
            if (dst_data.dependencyValues[dst_data.roundIndexToSend] != 0) {
              bitset_dependency.set(dst);
            }
          }
        }
      },
      galois::loopname(
          graph.get_run_identifier("BackFindMessageToSend")
              .c_str()),
      galois::no_stats());
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
            cur_data.bc += cur_data.dependencyValues[i];
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