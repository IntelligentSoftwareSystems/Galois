/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

constexpr static const char* const REGION_NAME = "PR_BC";

#include <iostream>

#include "galois/DistGalois.h"
#include "DistBenchStart.h"
#include "galois/DReducible.h"
#include "galois/runtime/Tracer.h"

/******************************************************************************/
/* Declaration of command line arguments */
/******************************************************************************/
namespace cll = llvm::cl;

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
static cll::opt<bool> noSkipSources("noSkipSources", 
                                   cll::desc("Do not skip sources without " 
                                             "outgoing edges (default skips)."),
                                   cll::init(false));
static cll::opt<unsigned int> vIndex("index", 
                                cll::desc("DEBUG: Index to print for dist/short "
                                          "paths"),
                                cll::init(0),
                                cll::Hidden);
static cll::opt<bool> outputDistPaths("outputDistPaths", 
                                      cll::desc("DEBUG: Output min distance"
                                                "/short path counts instead"),
                                      cll::init(false),
                                      cll::Hidden);

/******************************************************************************/
/* Graph structure declarations */
/******************************************************************************/
const uint32_t infinity = std::numeric_limits<uint32_t>::max() / 4;

// NOTE: declared types assume that these values will not reach uint64_t: it may
// need to be changed for very large graphs
struct NodeData {
  galois::gstl::Vector<uint32_t> oldMinDistances; // min distances in a previous round
  galois::gstl::Vector<uint32_t> minDistances; // current min distances for each source
  galois::gstl::Vector<uint64_t> shortestPathToAdd; // shortest path accumulator
  galois::gstl::Vector<uint64_t> shortestPathNumbers; // actual shortest path number
  galois::gstl::Vector<char> sentFlag; // marks if message has been sent for a source

  uint32_t APSPIndexToSend; // index that needs to be sent in a round

  // round numbers saved for determining when to send out back-prop messages
  galois::gstl::Vector<uint32_t> savedRoundNumbers; 

  // accumulator for adding to dependency values
  galois::gstl::Vector<galois::CopyableAtomic<float>> dependencyToAdd;
  // final dependency values
  galois::gstl::Vector<float> dependencyValues;

  uint64_t numFinalizedSources;

  float bc;
};

// Bitsets for tracking which nodes need to be sync'd with respect to a 
// particular field
#ifndef _VECTOR_SYNC_
std::vector<galois::DynamicBitSet> vbitset_minDistances;
std::vector<galois::DynamicBitSet> vbitset_shortestPathToAdd;
std::vector<galois::DynamicBitSet> vbitset_dependencyToAdd;
#else
galois::DynamicBitSet bitset_minDistances;
galois::DynamicBitSet bitset_shortestPathToAdd;
galois::DynamicBitSet bitset_dependencyToAdd;
#endif

// Dist Graph using a bidirectional CSR graph (3rd argument set to true does 
// this)
using Graph = galois::graphs::DistGraph<NodeData, void, true>;
using GNode = typename Graph::GraphNode;

#include "pr_bc_sync.hh"

/******************************************************************************/
/* Functions for running the algorithm */
/******************************************************************************/
uint64_t macroRound = 0; // macro round, i.e. number of batches done so far

/**
 * Graph initialization. Initialize all of the node data fields.
 *
 * @param graph Local graph to operate on
 */
void InitializeGraph(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();

  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode curNode) {
      NodeData& cur_data = graph.getData(curNode);
  
      cur_data.oldMinDistances.resize(numSourcesPerRound);
      cur_data.minDistances.resize(numSourcesPerRound);
      cur_data.shortestPathToAdd.resize(numSourcesPerRound);
      cur_data.shortestPathNumbers.resize(numSourcesPerRound);
  
      cur_data.APSPIndexToSend = numSourcesPerRound + 1;
  
      cur_data.savedRoundNumbers.resize(numSourcesPerRound);
      cur_data.sentFlag.resize(numSourcesPerRound);

      cur_data.dependencyToAdd.resize(numSourcesPerRound);
      cur_data.dependencyValues.resize(numSourcesPerRound);
  
      cur_data.numFinalizedSources = 0;

      cur_data.bc = 0.0;
    },
    galois::loopname(graph.get_run_identifier("InitializeGraph").c_str()), 
    galois::no_stats()
  );
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
    [&] (GNode curNode) {
      NodeData& cur_data = graph.getData(curNode);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        // min distance and short path count setup
        if (nodesToConsider[i] == graph.getGID(curNode)) {
          cur_data.minDistances[i] = 0;
          cur_data.shortestPathNumbers[i] = 1;
        } else {
          cur_data.minDistances[i] = infinity;
          cur_data.shortestPathNumbers[i] = 0;
        }
        cur_data.oldMinDistances[i] = cur_data.minDistances[i];

        cur_data.shortestPathToAdd[i] = 0;
        cur_data.sentFlag[i] = 0;
  
        cur_data.APSPIndexToSend = numSourcesPerRound;
  
        cur_data.savedRoundNumbers[i] = infinity;

        cur_data.dependencyToAdd[i] = 0.0;
        cur_data.dependencyValues[i] = 0.0;

        cur_data.numFinalizedSources = 0;
      }
    },
    galois::loopname(graph.get_run_identifier("InitializeIteration",
                                              macroRound).c_str()), 
    galois::no_stats()
  );
};


/**
 * If min distance has changed, then the number of shortest paths is reset
 * and the sent flag is reset.
 *
 * @param graph Local graph to operate on
 */
void MetadataUpdate(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();

  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode curNode) {
      NodeData& cur_data = graph.getData(curNode);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        if (cur_data.oldMinDistances[i] != cur_data.minDistances[i]) {
          cur_data.shortestPathNumbers[i] = 0;
          //cur_data.sentFlag[i] = 0; // reset sent flag
        }
      }
    },
    galois::loopname(graph.get_run_identifier("MetadataUpdate",
                                              macroRound).c_str()),
    galois::no_stats()
  );
};


// TODO
// This has dumb overhead even if nothing to update; find a way to make it
// faster
/**
 * Adds the accumulated shortest path values to the actual shortest path
 * value and resets the accumulator.
 *
 * @param graph Local graph to operate on
 */
void ShortPathUpdate(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();
  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode curNode) {
      NodeData& cur_data = graph.getData(curNode);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        if (cur_data.shortestPathToAdd[i] > 0) {
          cur_data.shortestPathNumbers[i] += cur_data.shortestPathToAdd[i];
        }
        cur_data.shortestPathToAdd[i] = 0;
      }
    },
    galois::loopname(graph.get_run_identifier("ShortPathUpdate",
                                              macroRound).c_str()),
    galois::no_stats()
  );
};


/**
 * Update old dist with the latest value of the min distance.
 *
 * @param graph Local graph to operate on
 */
void OldDistUpdate(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();

  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode curNode) {
      NodeData& cur_data = graph.getData(curNode);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        cur_data.oldMinDistances[i] = cur_data.minDistances[i];
      }
    },
    galois::loopname(graph.get_run_identifier("OldDistUpdate",
                                              macroRound).c_str()),
    galois::no_stats()
  );
};


/**
 * Determine if a node needs to send out a shortest path message in this
 * round and saves it to the node data struct for later use.
 *
 * @param graph Local graph to operate on
 * @param roundNumber current round number
 * @param dga Distributed accumulator for determining if work was done in
 * an iteration across all hosts
 */
void FindMessageToSend(Graph& graph, const uint32_t roundNumber, 
                       galois::DGAccumulator<uint32_t>& dga) {
  const auto& allNodes = graph.allNodesRange(); 

  galois::do_all(
    galois::iterate(allNodes), 
    [&] (GNode curNode) {
      NodeData& cur_data = graph.getData(curNode);

      bool continueWork = false;

      cur_data.APSPIndexToSend = numSourcesPerRound;
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        if (((cur_data.numFinalizedSources + cur_data.oldMinDistances[i]) == 
            roundNumber) && !cur_data.sentFlag[i]) {
          cur_data.savedRoundNumbers[i] = roundNumber; // safe
          cur_data.APSPIndexToSend = i;
          assert(cur_data.shortestPathNumbers[i] != 0);
          cur_data.numFinalizedSources++;
          cur_data.sentFlag[i] = 1; // reset sent flag
          continueWork = true;
          break;
        } else if (cur_data.oldMinDistances[i] != infinity && 
                   !cur_data.sentFlag[i]) {
          continueWork = true;
        }
      }

      if (continueWork) {
        dga += 1;
      }
    },
    galois::loopname(graph.get_run_identifier("FindMessageToSend",
                                              macroRound).c_str()),
    galois::no_stats(),
    galois::steal()
  );
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
  const auto& allNodesWithEdgesIn = graph.allNodesWithEdgesRangeIn();

  galois::do_all(
    galois::iterate(allNodesWithEdgesIn),
    [&] (GNode dst) {
      auto& dnode = graph.getData(dst);

      for (auto inEdge : graph.in_edges(dst)) {
        NodeData& src_data = graph.getData(graph.getInEdgeDst(inEdge));
        uint32_t indexToSend = src_data.APSPIndexToSend;
        
        if (indexToSend != numSourcesPerRound) {
          uint32_t distValue = src_data.oldMinDistances[indexToSend];
          uint32_t newValue = distValue + 1;
          uint32_t oldValue = galois::min(dnode.minDistances[indexToSend],
                                          newValue);

          assert(src_data.shortestPathNumbers[indexToSend] != 0);

          if (oldValue > newValue) {
            // overwrite short path with this node's shortest path
            dnode.shortestPathToAdd[indexToSend] = 
              src_data.shortestPathNumbers[indexToSend];
          } else if (oldValue == newValue) {
            // add to short path
            dnode.shortestPathToAdd[indexToSend] += 
              src_data.shortestPathNumbers[indexToSend];
          }

          dga += 1;
          #ifndef _VECTOR_SYNC_
          vbitset_minDistances[indexToSend].set(dst);
          vbitset_shortestPathToAdd[indexToSend].set(dst);
          #else
          bitset_minDistances.set(dst);
          bitset_shortestPathToAdd.set(dst);
          #endif
        }
      }
    },
    galois::loopname(graph.get_run_identifier("SendAPSPMessages", 
                                              macroRound).c_str()),
    galois::no_stats(),
    galois::steal()
  );
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

    // find the message a node needs to send (if any)
    FindMessageToSend(graph, roundNumber, dga); 
    // send messages (if any)
    SendAPSPMessages(graph, dga);

    // sync min distance (also resets shortPathAdd if necessary)
    graph.sync<writeDestination, readAny, ReducePairwiseMinAndResetDist, 
               Broadcast_minDistances, 
               Bitset_minDistances>(std::string("MinDistSync") + "_" + 
                                    std::to_string(macroRound));

    // updates short path count and the sent flag based on results of this
    // round's APSP
    MetadataUpdate(graph); 

    // sync shortPathAdd
    #ifndef _VECTOR_SYNC_
    graph.sync<writeDestination, readAny, 
               Reduce_pair_wise_add_array_single_shortestPathToAdd, 
               Broadcast_shortestPathToAdd, 
               Bitset_shortestPathToAdd>(std::string("ShortPathSync") + "_" +
                                         std::to_string(macroRound));
    #else
    graph.sync<writeDestination, readAny, 
               Reduce_pair_wise_add_array_shortestPathToAdd, 
               Broadcast_shortestPathToAdd, 
               Bitset_shortestPathToAdd>(std::string("ShortPathSync") + "_" +
                                         std::to_string(macroRound));
    #endif

    // update short path count with sync'd accumulator
    ShortPathUpdate(graph);

    // old dist gets updated with new dist
    OldDistUpdate(graph);

    roundNumber++;
  } while (dga.reduce());

  return roundNumber;
}


/**
 * Get the round number for the backward propagation phase using the round
 * number from the APSP phase. This round number determines when a node should
 * send out a message for the backward propagation of dependency values.
 *
 * @param graph Local graph to operate on
 * @param lastRoundNumber last round number in the APSP phase
 */
void RoundUpdate(Graph& graph, const uint32_t lastRoundNumber) {
  const auto& allNodes = graph.allNodesRange();
  graph.set_num_round(0);

  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode src) {
      NodeData& src_data = graph.getData(src);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        if (src_data.oldMinDistances[i] < infinity) {
          src_data.savedRoundNumbers[i] = lastRoundNumber - 
                                          src_data.savedRoundNumbers[i];
          assert(src_data.savedRoundNumbers[i] <= lastRoundNumber);
        }
      }
    },
    galois::loopname(graph.get_run_identifier("RoundUpdate", 
                                              macroRound).c_str()),
    galois::no_stats()
  );
}


/**
 * Add dependency accumulator to final dependency value.
 *
 * @param graph Local graph to operate on
 */
void UpdateDependency(Graph& graph) {
  const auto& allNodes = graph.allNodesRange();

  galois::do_all(
    galois::iterate(allNodes.begin(), allNodes.end()), 
    [&] (GNode curNode) {
      NodeData& cur_data = graph.getData(curNode);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        if (cur_data.dependencyToAdd[i] > 0) {
          cur_data.dependencyValues[i] += cur_data.dependencyToAdd[i].load();
        }
        cur_data.dependencyToAdd[i] = 0;
      }
    },
    galois::loopname(graph.get_run_identifier("UpdateDependency",
                                              macroRound).c_str()),
    galois::no_stats()
  );
}


/**
 * Back propagate dependency values depending on the round that a node
 * sent out the shortest path message.
 *
 * @param graph Local graph to operate on
 * @param lastRoundNumber last round number in the APSP phase
 */
void BackProp(Graph& graph, const uint32_t lastRoundNumber) {
  const auto& allNodesWithEdgesIn = graph.allNodesWithEdgesRangeIn();

  uint32_t currentRound = 0;

  while (currentRound <= lastRoundNumber) {
    graph.set_num_round(currentRound);

    galois::do_all(
      galois::iterate(allNodesWithEdgesIn),
      [&] (GNode dst) {
        NodeData& dst_data = graph.getData(dst);
    
        std::vector<uint32_t> toBackProp;

        for (unsigned i = 0; i < numSourcesPerRound; i++) {
          if (dst_data.savedRoundNumbers[i] == currentRound) {
            toBackProp.emplace_back(i);
            break;
          }
        }
    
        assert(toBackProp.size() <= 1);
    
        for (auto i : toBackProp) {
          uint32_t myDistance = dst_data.oldMinDistances[i];
    
          // calculate final dependency value
          dst_data.dependencyValues[i] = dst_data.dependencyValues[i] * 
                                         dst_data.shortestPathNumbers[i];
    
          // get the value to add to predecessors
          float toAdd = ((float)1 + dst_data.dependencyValues[i]) / 
                          dst_data.shortestPathNumbers[i];
    
          for (auto inEdge : graph.in_edges(dst)) {
            GNode src = graph.getInEdgeDst(inEdge);
            auto& src_data = graph.getData(src);
    
            // determine if this source is a predecessor
            if (myDistance == (src_data.oldMinDistances[i] + 1)) {
              // add to dependency of predecessor using our finalized one
              galois::atomicAdd(src_data.dependencyToAdd[i], toAdd);

              #ifndef _VECTOR_SYNC_
              vbitset_dependencyToAdd[i].set(src); 
              #else
              bitset_dependencyToAdd.set(src); 
              #endif
            }
          }
        }
      },
      galois::loopname(graph.get_run_identifier("BackProp",
                                                macroRound).c_str()),
      galois::steal(),
      galois::no_stats()
    );

    // TODO can be possibly optimized? (see comment below)
    #ifndef _VECTOR_SYNC_
    graph.sync<writeSource, readAny, 
               Reduce_pair_wise_add_array_single_dependencyToAdd, 
               Broadcast_dependencyToAdd, 
               Bitset_dependencyToAdd>(std::string("DependencySync") + "_" +
                                       std::to_string(macroRound));
    #else
    graph.sync<writeSource, readAny, 
               Reduce_pair_wise_add_array_dependencyToAdd, 
               Broadcast_dependencyToAdd, 
               Bitset_dependencyToAdd>(std::string("DependencySync") + "_" +
                                       std::to_string(macroRound));
    #endif

    // dependency written to source
    // dep needs to be on dst nodes, but final round needs them on all nodes

    UpdateDependency(graph);

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
    [&] (GNode node) {
      NodeData& cur_data = graph.getData(node);
  
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        // exclude sources themselves from BC calculation
        if (graph.getGID(node) != nodesToConsider[i]) {
          cur_data.bc += cur_data.dependencyValues[i];
        }

      }
    },
    galois::loopname(graph.get_run_identifier("BC", macroRound).c_str()),
    galois::no_stats()
  );
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

  galois::do_all(
    galois::iterate(graph.masterNodesRange().begin(), 
                    graph.masterNodesRange().end()),
    [&] (auto src) {
      NodeData& sdata = graph.getData(src);

      DGA_max.update(sdata.bc);
      DGA_min.update(sdata.bc);
      DGA_sum += sdata.bc;
    },
    galois::no_stats(),
    galois::loopname("Sanity")
  );

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

constexpr static const char* const name = "Pontecorvi-Ramachandran Betweeness "
                                          "Centrality"; 
constexpr static const char* const desc = "Pontecorvi-Ramachandran Betweeness "
                                          "Centrality on Distributed Galois.";
constexpr static const char* const url = 0;

int main(int argc, char** argv) {
  galois::DistMemSys G;
  DistBenchStart(argc, argv, name, desc, url);

  auto& net = galois::runtime::getSystemNetworkInterface();

  galois::StatTimer StatTimer_total("TimerTotal", REGION_NAME);
  StatTimer_total.start();

  Graph* hg = twoWayDistGraphInitialization<NodeData, void>();

  galois::gPrint("[", net.ID, "] InitializeGraph\n");

  galois::StatTimer StatTimer_graph_init("TIMER_GRAPH_INIT", REGION_NAME);

  StatTimer_graph_init.start();
  InitializeGraph(*hg);
  StatTimer_graph_init.stop();

  galois::runtime::getHostBarrier().wait();

  // shared DG accumulator among all steps
  galois::DGAccumulator<uint32_t> dga;

  if (totalNumSources == 0) {
    galois::gDebug("Total num sources unspecified");
    totalNumSources = hg->globalSize();
  }

  if (useSingleSource) {
    totalNumSources = 1;
    numSourcesPerRound = 1;
  }

  // bitset initialization
  #ifndef _VECTOR_SYNC_
  vbitset_minDistances.resize(numSourcesPerRound);
  vbitset_shortestPathToAdd.resize(numSourcesPerRound);
  vbitset_dependencyToAdd.resize(numSourcesPerRound);

  for (unsigned i = 0; i < numSourcesPerRound; i++) {
    vbitset_minDistances[i].resize(hg->size());
    vbitset_shortestPathToAdd[i].resize(hg->size());
    vbitset_dependencyToAdd[i].resize(hg->size());
  }
  #else
  bitset_minDistances.resize(hg->size());
  bitset_shortestPathToAdd.resize(hg->size());
  bitset_dependencyToAdd.resize(hg->size());
  #endif

  uint64_t origNumRoundSources = numSourcesPerRound;

  std::vector<uint64_t> nodesToConsider;
  nodesToConsider.resize(numSourcesPerRound);

  for (auto run = 0; run < numRuns; ++run) {
    galois::gPrint("[", net.ID, "] Run ", run, " started\n");
    std::string timer_str("Timer_" + std::to_string(run));
    galois::StatTimer StatTimer_main(timer_str.c_str(), REGION_NAME);

    // offset into sources to operate on
    uint64_t offset = 0;
    uint64_t numNodes = hg->globalSize();
    uint64_t totalSourcesFound = 0;

    galois::DGAccumulator<unsigned> hasEdges;
    while (offset < numNodes && totalSourcesFound < totalNumSources) {
      unsigned sourcesFound = 0;
      
      while (sourcesFound < numSourcesPerRound && offset < numNodes &&
             totalSourcesFound < totalNumSources) {
        if (!noSkipSources) {
          hasEdges.reset();

          // find out if this node is local + has an outgoing edge
          if (hg->isLocal(offset)) {
            unsigned localID = hg->G2L(offset);

            if (std::distance(hg->edge_begin(localID), hg->edge_end(localID))) {
              hasEdges += 1;
            }
          }

          // if this node has an outgoing edge on any node, add it to vector of 
          // sources to consider
          if (hasEdges.reduce()) {
            nodesToConsider[sourcesFound] = offset;
            sourcesFound++;
            totalSourcesFound++;
          } else {
            if (net.ID == 0) {
              galois::gDebug("Skipping node ", offset, " (no outgoing edges)");
            }
          }
        } else {
          // no skip
          nodesToConsider[sourcesFound] = offset;
          sourcesFound++;
          totalSourcesFound++;
        }

        offset++;
      }

      if (sourcesFound == 0) {
        assert(offset == totalNumSources || 
               totalSourcesFound == totalNumSources);
        break;
      }

      // correct numSourcesPerRound if not enough sources found
      if (offset < totalNumSources) {
        assert(numSourcesPerRound == sourcesFound);
      } else {
        galois::gDebug("Not enough sources found (only found ", sourcesFound, 
                       ")");
        numSourcesPerRound = sourcesFound;
      }

      if (useSingleSource) {
        nodesToConsider[0] = startNode;
      }

      StatTimer_main.start();
      InitializeIteration(*hg, nodesToConsider);

      // APSP returns total number of rounds taken
      // subtract 1 to get to terminating round; i.e. last round 
      uint32_t lastRoundNumber = APSP(*hg, dga) - 1;

      RoundUpdate(*hg, lastRoundNumber);
      BackProp(*hg, lastRoundNumber);
      BC(*hg, nodesToConsider);
      StatTimer_main.stop();

      macroRound++;
    }

    // sanity 
    Sanity(*hg);

    // re-init graph for next run
    if ((run + 1) != numRuns) {
      galois::runtime::getHostBarrier().wait();
      (*hg).set_num_run(run + 1);
      (*hg).set_num_round(0);
      offset = 0;
      macroRound = 0;
      numSourcesPerRound = origNumRoundSources;

      #ifndef _VECTOR_SYNC_
      for (unsigned i = 0; i < numSourcesPerRound; i++) {
        vbitset_minDistances[i].reset();
        vbitset_shortestPathToAdd[i].reset();
        vbitset_dependencyToAdd[i].reset();
      }
      #else
      bitset_minDistances.reset();
      bitset_shortestPathToAdd.reset();
      bitset_dependencyToAdd.reset();
      #endif

      InitializeGraph(*hg);
      galois::runtime::getHostBarrier().wait();
    }
  }

  StatTimer_total.stop();

  // Verify, i.e. print out graph data for examination
  if (verify) {
    // buffer for text to be written out to file
    char* v_out = (char*)malloc(40); 

    for (auto ii = (*hg).masterNodesRange().begin(); 
              ii != (*hg).masterNodesRange().end(); 
              ++ii) {
      if (!outputDistPaths) {
        // outputs betweenness centrality
        sprintf(v_out, "%lu %.9f\n", (*hg).getGID(*ii),
                                     (*hg).getData(*ii).bc);
      } else {
        // outputs min distance and short path numbers
        sprintf(v_out, "%lu %u %lu\n", (*hg).getGID(*ii),
                                      (*hg).getData(*ii).minDistances[vIndex],
                                      (*hg).getData(*ii).shortestPathNumbers[vIndex]);
      }

      galois::runtime::printOutput(v_out);
    }

    free(v_out);
  }

  return 0;
}
