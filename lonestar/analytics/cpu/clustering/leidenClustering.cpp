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

#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"
#include "galois/AtomicHelpers.h"

#include <iostream>
#include <fstream>
#include <deque>
#include <type_traits>

#include "Lonestar/BoilerPlate.h"
#include "clustering.h"
#include "galois/DynamicBitset.h"

static const char* name = "Louvain Clustering";

static const char* desc = "Cluster nodes of the graph using Louvain Clustering";

static const char* url = "louvain_clustering";

enum Algo { foreach };

static cll::opt<std::string>
    filename(cll::Positional, cll::desc("<input graph>"), cll::Required);

static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm:"),
    cll::values(clEnumValN(Algo::foreach, "Foreach",
                           "Using galois for_each for conflict mitigation")),
    cll::init(Algo::foreach));

// Maintain community information
struct Comm {
  std::atomic<uint64_t> size;
  std::atomic<EdgeTy> degree_wt;
  std::atomic<uint64_t> node_wt;
  EdgeTy internal_edge_wt;
  uint64_t num_subcomm;
};

typedef galois::LargeArray<Comm> CommArray;
// Graph Node information
struct Node {
  uint64_t prev_comm_ass;
  uint64_t curr_comm_ass;
  EdgeTy degree_wt;
  int64_t colorId;
  /** Only required for Leiden **/
  uint64_t curr_subcomm_ass;
  uint64_t node_wt;
};

using Graph = galois::graphs::LC_CSR_Graph<Node, EdgeTy>::with_no_lockable<
    false>::type::with_numa_alloc<true>::type;
using GNode = Graph::GraphNode;

double algoLeidenWithLocking(Graph& graph, double lower, double threshold,
                             uint32_t& iter) {

  galois::StatTimer TimerClusteringTotal("Timer_Clustering_Total");
  TimerClusteringTotal.start();

  galois::gPrint("Inside algoLeidenWithLocking\n");

  CommArray c_info;   // Community info
  CommArray c_update; // Used for updating community

  /* Variables needed for Modularity calculation */
  double constant_for_second_term;
  double prev_mod      = lower;
  double curr_mod      = -1;
  double threshold_mod = threshold;
  uint32_t num_iter    = iter;

  /*** Initialization ***/
  c_info.allocateBlocked(graph.size());
  c_update.allocateBlocked(graph.size());

  /* Calculate the weighted degree sum for each vertex */
  sumVertexDegreeWeightWithNodeWeight(graph, c_info);

  /* Compute the total weight (2m) and 1/2m terms */
  constant_for_second_term = calConstantForSecondTerm(graph);

  if (iter > 1) {
    galois::do_all(galois::iterate(graph), [&](GNode n) {
      c_info[n].size      = 0;
      c_info[n].degree_wt = 0;
      c_info[n].node_wt   = 0;
    });

    galois::do_all(galois::iterate(graph), [&](GNode n) {
      auto& n_data = graph.getData(n);
      galois::atomicAdd(c_info[n_data.curr_comm_ass].size, (uint64_t)1);
      galois::atomicAdd(c_info[n_data.curr_comm_ass].node_wt, n_data.node_wt);
      galois::atomicAdd(c_info[n_data.curr_comm_ass].degree_wt,
                        n_data.degree_wt);
    });
  }

  galois::gPrint("============================================================="
                 "===========================================\n");
  galois::gPrint("Itr      Explore_xx            A_x2          Prev-Prev-Mod   "
                 "      Prev-Mod           Curr-Mod\n");
  galois::gPrint("============================================================="
                 "===========================================\n");

  galois::StatTimer TimerClusteringWhile("Timer_Clustering_While");
  TimerClusteringWhile.start();
  while (true) {
    num_iter++;
    galois::do_all(galois::iterate(graph), [&](GNode n) {
      c_update[n].degree_wt = 0;
      c_update[n].size      = 0;
      c_update[n].node_wt   = 0;
    });

    galois::for_each(
        galois::iterate(graph),
        [&](GNode n, auto&) {
          auto& n_data    = graph.getData(n, flag_write_lock);
          uint64_t degree = std::distance(graph.edge_begin(n, flag_write_lock),
                                          graph.edge_end(n, flag_write_lock));

          uint64_t local_target = UNASSIGNED;
          std::map<uint64_t, uint64_t>
              cluster_local_map; // Map each neighbor's cluster to local number:
                                 // Community --> Index
          std::vector<EdgeTy> counter; // Number of edges to each unique cluster
          EdgeTy self_loop_wt = 0;

          if (degree > 0) {
            findNeighboringClusters(graph, n, cluster_local_map, counter,
                                    self_loop_wt);
            local_target =
                maxModularity(cluster_local_map, counter, self_loop_wt, c_info,
                              n_data.degree_wt, n_data.curr_comm_ass,
                              constant_for_second_term);
            // local_target = maxCPMQuality<Graph, CommArray>(cluster_local_map,
            // counter, self_loop_wt, c_info, n_data.node_wt,
            // n_data.curr_comm_ass);
          } else {
            local_target = UNASSIGNED;
          }

          /* Update cluster info */
          if (local_target != n_data.curr_comm_ass &&
              local_target != UNASSIGNED) {

            galois::atomicAdd(c_info[local_target].degree_wt, n_data.degree_wt);
            galois::atomicAdd(c_info[local_target].size, (uint64_t)1);
            galois::atomicAdd(c_info[local_target].node_wt, n_data.node_wt);

            galois::atomicSubtract(c_info[n_data.curr_comm_ass].degree_wt,
                                   n_data.degree_wt);
            galois::atomicSubtract(c_info[n_data.curr_comm_ass].size,
                                   (uint64_t)1);
            galois::atomicSubtract(c_info[n_data.curr_comm_ass].node_wt,
                                   n_data.node_wt);

            /* Set the new cluster id */
            n_data.curr_comm_ass = local_target;
          }
        },
        galois::loopname("leiden algo: Phase 1"), galois::no_pushes());

    /* Calculate the overall modularity */
    double e_xx = 0;
    double a2_x = 0;

    // curr_mod = calCPMQuality(graph, c_info, e_xx, a2_x,
    // constant_for_second_term);
    curr_mod =
        calModularity(graph, c_info, e_xx, a2_x, constant_for_second_term);

    galois::gPrint(num_iter, "        ", e_xx, "        ", a2_x, "        ",
                   lower, "      ", prev_mod, "       ", curr_mod, "\n");

    if ((curr_mod - prev_mod) < threshold_mod) {
      galois::gPrint("Modularity gain: ", (curr_mod - prev_mod), " < ",
                     threshold_mod, " \n");
      prev_mod = curr_mod;
      break;
    }
    prev_mod = curr_mod;
  } // End while
  TimerClusteringWhile.stop();

  iter = num_iter;

  c_info.destroy();
  c_info.deallocate();

  c_update.destroy();
  c_update.deallocate();

  TimerClusteringTotal.stop();
  return prev_mod;
}

void runMultiPhaseLouvainAlgorithm(Graph& graph, uint64_t min_graph_size,
                                   double c_threshold,
                                   largeArray& clusters_orig) {

  galois::gPrint("Inside runMultiPhaseLouvainAlgorithm\n");
  double prev_mod = -1; // Previous modularity
  double curr_mod = -1; // Current modularity
  uint32_t phase  = 0;

  Graph* graph_curr = &graph;
  Graph graph_next;
  uint32_t iter           = 0;
  uint64_t num_nodes_orig = clusters_orig.size();
  /**
   * Assign cluster id from previous iteration
   */
  galois::do_all(galois::iterate(*graph_curr), [&](GNode n) {
    graph_curr->getData(n).curr_comm_ass    = n;
    graph_curr->getData(n).curr_subcomm_ass = n;
    graph_curr->getData(n).node_wt          = 1;
  });
  for (GNode i = 0; i < graph.size(); ++i) {
    if (graph.getData(i).node_wt > 1)
      galois::gPrint("-->node wt : ", graph.getData(i).node_wt, "\n");
  }
  while (true) {
    iter++;
    phase++;
    galois::gPrint("Starting Phase : ", phase, "\n");
    galois::gPrint("Graph size : ", (*graph_curr).size(), "\n");

    if ((*graph_curr).size() > min_graph_size) {
      switch (algo) {
      case foreach:
        curr_mod =
            algoLeidenWithLocking(*graph_curr, curr_mod, c_threshold, iter);
        break;
      default:
        std::abort();
      }
    }

    if (iter < max_iter && (curr_mod - prev_mod) > threshold) {
      double constant_for_second_term = calConstantForSecondTerm(graph);
      refinePartition<Graph, CommArray>(*graph_curr, constant_for_second_term);

      uint64_t num_unique_subclusters =
          renumberClustersContiguouslySubcomm(*graph_curr);
      galois::gPrint("Number of unique sub cluster (Refine) : ",
                     num_unique_subclusters, "\n");
      std::vector<uint64_t> original_comm_ass(graph_curr->size());
      std::vector<uint64_t> cluster_node_wt(num_unique_subclusters, 0);

      if (phase == 1) {
        galois::do_all(
            galois::iterate((uint64_t)0, num_nodes_orig), [&](GNode n) {
              clusters_orig[n] = (*graph_curr).getData(n).curr_subcomm_ass;
            });
      } else {
        galois::do_all(
            galois::iterate((uint64_t)0, num_nodes_orig),
            [&](GNode n) {
              assert(clusters_orig[n] < (*graph_curr).size());
              clusters_orig[n] =
                  (*graph_curr).getData(clusters_orig[n]).curr_subcomm_ass;
            },
            galois::steal());
      }
      buildNextLevelGraphSubComm(*graph_curr, graph_next,
                                 num_unique_subclusters, original_comm_ass,
                                 cluster_node_wt);
      prev_mod   = curr_mod;
      graph_curr = &graph_next;
      /**
       * Assign cluster id from previous iteration
       */
      galois::do_all(galois::iterate(*graph_curr), [&](GNode n) {
        auto& n_data            = graph_curr->getData(n);
        n_data.curr_comm_ass    = original_comm_ass[n];
        n_data.curr_subcomm_ass = original_comm_ass[n];
        n_data.node_wt          = cluster_node_wt[n];
      });

      cluster_node_wt.clear();
      printGraphCharateristics(*graph_curr);
    } else {
      break;
    }
  }
  galois::gPrint("Phases : ", phase, "Iter : ", iter, "\n");
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  Graph graph, graph_next;
  Graph* graph_curr;

  galois::StatTimer TEnd2End("Timer_end2end");
  TEnd2End.start();

  std::cout << "Reading from file: " << filename << std::endl;
  std::cout << "[WARNING:] Make sure " << filename
            << " is symmetric graph without duplicate edges" << std::endl;
  galois::graphs::readGraph(graph, filename);
  std::cout << "Read " << graph.size() << " nodes, " << graph.sizeEdges()
            << " edges" << std::endl;

  graph_curr = &graph;

  /*
   * To keep track of communities for nodes in the original graph.
   * Community will be set to -1 for isolated nodes
   */
  largeArray clusters_orig;
  clusters_orig.allocateBlocked(graph_curr->size());

  /*
   * Vertex following optimization
   */
  if (enable_VF) {
    uint64_t num_nodes_to_fix =
        vertexFollowing(graph); // Find nodes that follow other nodes
    galois::gPrint("Isolated nodes : ", num_nodes_to_fix, "\n");

    uint64_t num_unique_clusters = renumberClustersContiguously(*graph_curr);
    galois::gPrint(
        "Number of unique clusters (renumber): ", num_unique_clusters, "\n");
    /*
     *Initialize node cluster id.
     */
    galois::do_all(galois::iterate(*graph_curr), [&](GNode n) {
      clusters_orig[n] = graph.getData(n, flag_no_lock).curr_comm_ass;
    });

    /*
     * Build new graph to remove the isolated nodes
     */
    buildNextLevelGraph(*graph_curr, graph_next, num_unique_clusters);
    graph_curr = &graph_next;
    printGraphCharateristics(*graph_curr);
  } else {

    /*
     *Initialize node cluster id.
     */
    galois::do_all(galois::iterate(*graph_curr),
                   [&](GNode n) { clusters_orig[n] = UNASSIGNED; });

    printGraphCharateristics(*graph_curr);
  }

  uint64_t min_graph_size = 10;
  galois::StatTimer Tmain("Timer_LC");
  Tmain.start();
  runMultiPhaseLouvainAlgorithm(*graph_curr, min_graph_size, c_threshold,
                                clusters_orig);
  Tmain.stop();

  TEnd2End.stop();

  /*
   * Sanity check: Check modularity at the end
   */
  checkModularity<Graph, CommArray>(graph, clusters_orig);
  if (output_CID) {
    printNodeClusterId(graph, output_CID_filename);
  }
  return 0;
}
