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

#include "louvain-partitioning.h"
#include "galois/Galois.h"
#include "galois/AtomicHelpers.h"
#include "galois/DynamicBitset.h"
#include "galois/gstl.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "Lonestar/BoilerPlate.h"

#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <fstream>
#include <type_traits>

static const char* name = "Partitioning using Louvain Clustering";

static const char* desc =
    "Partition nodes of the graph using Louvain Clustering";

static const char* url = "louvain_partitioning";

namespace cll = llvm::cl;

static cll::opt<std::string> inputFile("input", cll::desc("<input file>"),
                                       cll::Required);

static cll::opt<double> tolerance("tolerance",
                                  cll::desc("Percentage deviation (default 5)"),
                                  cll::init(5));

static cll::opt<double> c_threshold("c_threshold",
                                    cll::desc("Threshold for modularity gain"),
                                    cll::init(0.01));

static cll::opt<double>
    threshold("threshold", cll::desc("Total threshold for modularity gain"),
              cll::init(0.01));

static cll::opt<uint32_t>
    max_iter("max_iter", cll::desc("Maximum number of iterations to execute"),
             cll::init(10));

static cll::opt<bool>
    output_CID("output_CID", cll::desc("Flag to enable cluster ID printing."),
               cll::init(false));

static cll::opt<std::string>
    output_CID_filename("output_CID_filename",
                        cll::desc("File name to output cluster IDs."),
                        cll::init("output_CID_filename"));

static cll::opt<uint32_t>
    min_graph_size("min_graph_size", cll::desc("Minimum coarsened graph size"),
                   cll::init(100));

// Maintain community information
struct Comm {
  std::atomic<uint64_t> size;
  std::atomic<uint64_t> degree_wt;
};

typedef galois::LargeArray<Comm> CommArray;

double algoLouvainCoarsening(Graph& graph, double lower, double threshold,
                             uint32_t& iter) {
  galois::gPrint("Inside algoLouvainCoarsening\n");

  galois::StatTimer TimerClusteringTotal("Timer_Clustering_Total");
  TimerClusteringTotal.start();

  CommArray c_info;   // Community info
  CommArray c_update; // Used for updating community

  double constant_for_second_term;
  double prev_mod      = -1; // lower;
  double curr_mod      = -1;
  double threshold_mod = threshold;
  uint32_t num_iter    = iter;

  c_info.allocateBlocked(graph.size());
  c_update.allocateBlocked(graph.size());

  galois::do_all(galois::iterate(graph), [&graph](GNode n) {
    graph.getData(n).curr_comm_ass = n;
    graph.getData(n).prev_comm_ass = n;
  });

  sumVertexDegreeWeight(graph, c_info);

  constant_for_second_term = calConstantForSecondTerm(graph);

  galois::StatTimer TimerClusteringWhile("Timer_Clustering_While");
  TimerClusteringWhile.start();

  std::vector<uint64_t> local_target(graph.size(), UNASSIGNED);

  // partition nodes
  std::vector<galois::InsertBag<GNode>> bag(16);

  galois::InsertBag<GNode> toProcess;

  galois::LargeArray<bool> inBag;

  inBag.allocateBlocked(graph.size());

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    int64_t idx = n % 16;
    bag[idx].push(n);

    inBag[n] = false;
  });

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    c_update[n].degree_wt = 0;
    c_update[n].size      = 0;
  });

  while (true) {
    num_iter++;

    for (int64_t idx = 0; idx <= 15; idx++) {

      galois::do_all(
          galois::iterate(bag[idx]),
          [&](GNode n) {
            auto& n_data    = graph.getData(n, flag_write_lock);
            uint64_t degree = std::distance(graph.edge_begin(n, flag_no_lock),
                                            graph.edge_end(n, flag_no_lock));

            std::map<uint64_t, uint64_t>
                cluster_local_map; // Map each neighbor's cluster to local
                                   // number: Community --> Index
                                   //
            std::vector<EdgeTy>
                counter; // Number of edges to each unique cluster

            uint64_t num_unique_clusters = 1;
            uint64_t self_loop_wt        = 0;

            if (degree > 0) {

              cluster_local_map[graph.getData(n).curr_comm_ass] =
                  0; // Add n's current cluster
              EdgeTy edge;
              edge.weight = 0;
              counter.push_back(edge);

              for (auto ii = graph.edge_begin(n); ii != graph.edge_end(n);
                   ++ii) {
                GNode dst = graph.getEdgeDst(ii);

                auto edge_wt = graph.getEdgeData(
                    ii, flag_no_lock); // Self loop weights is recorded
                if (dst == n) {
                  self_loop_wt +=
                      edge_wt.weight; // Self loop weights is recorded
                }
                auto stored_already = cluster_local_map.find(
                    graph.getData(dst)
                        .curr_comm_ass); // Check if it already exists
                if (stored_already != cluster_local_map.end()) {
                  counter[stored_already->second].weight += edge_wt.weight;
                } else {
                  cluster_local_map[graph.getData(dst).curr_comm_ass] =
                      num_unique_clusters;
                  counter.push_back(edge_wt);
                  num_unique_clusters++;
                }
              } // End edge loop

              local_target[n] =
                  maxEdgeCut(cluster_local_map, counter, self_loop_wt, c_info,
                             n_data.curr_comm_ass, constant_for_second_term);
            } else {
              local_target[n] = n_data.curr_comm_ass;
            }

            if (local_target[n] != n_data.curr_comm_ass &&
                local_target[n] != UNASSIGNED) {
              galois::atomicAdd(c_update[local_target[n]].degree_wt,
                                n_data.degree_wt);
              galois::atomicAdd(c_update[local_target[n]].size, (uint64_t)1);
              galois::atomicSubtract(c_update[n_data.curr_comm_ass].degree_wt,
                                     n_data.degree_wt);
              galois::atomicSubtract(c_update[n_data.curr_comm_ass].size,
                                     (uint64_t)1);

              if (!inBag[local_target[n]]) {
                toProcess.push(local_target[n]);
                inBag[local_target[n]] = true;
              }

              if (!inBag[n_data.curr_comm_ass]) {
                toProcess.push(n_data.curr_comm_ass);
                inBag[n_data.curr_comm_ass] = true;
              }
            }
          },
          galois::loopname("louvain coarsening algo"));

      galois::do_all(galois::iterate(bag[idx]), [&](GNode n) {
        auto& n_data         = graph.getData(n, flag_no_lock);
        n_data.prev_comm_ass = n_data.curr_comm_ass;
        n_data.curr_comm_ass = local_target[n];
      });

      for (auto n : toProcess) {
        if (inBag[n]) {
          c_info[n].size += c_update[n].size.load();
          c_info[n].degree_wt += c_update[n].degree_wt.load();

          c_update[n].size      = 0;
          c_update[n].degree_wt = 0;
          inBag[n]              = false;
        }
      }
      toProcess.clear();
    } // end for

    double e_xx = 0;

    curr_mod =
        calModularityDelay(graph, e_xx, constant_for_second_term, local_target);

    if ((curr_mod - prev_mod) < threshold_mod) {
      prev_mod = curr_mod;

      break;
    }

    prev_mod = curr_mod;
    if (prev_mod < lower)
      prev_mod = lower;

  } // End while

  TimerClusteringWhile.stop();

  iter = num_iter;

  c_info.destroy();
  c_info.deallocate();

  c_update.destroy();
  c_update.deallocate();

  inBag.destroy();
  inBag.deallocate();

  TimerClusteringTotal.stop();
  return prev_mod;
}

void runMultiLevelAlgorithm(Graph& graph, uint32_t min_graph_size,
                            double c_threshold) {

  galois::gPrint("Inside runMultiLevelAlgorithm\n");
  uint32_t phase = 0;

  Graph* graph_curr = &graph;
  Graph graph_next[20];

  int l         = 0;
  uint32_t iter = 0;

  double curr_mod = -1;

  std::vector<Graph*> graphs;
  graphs.push_back(&graph);

  // Coarsening Phase
  while (true) {
    l++;
    iter++;
    phase++;
    galois::gPrint("Starting Phase : ", phase, "\n");
    galois::gPrint("Graph size : ", (*graph_curr).size(), "\n");

    computeDegrees(*graph_curr);

    if ((*graph_curr).size() > min_graph_size) {

      curr_mod =
          algoLouvainCoarsening(*graph_curr, curr_mod, c_threshold, iter);
    }

    uint64_t num_unique_clusters = renumberClustersContiguously(*graph_curr);

    if (num_unique_clusters < graph_curr->size() && phase < 10) {
      buildNextLevelGraph(*graph_curr, graph_next[l], num_unique_clusters);
      graph_curr = &graph_next[l];
      printGraphCharateristics(*graph_curr);
      graphs.push_back(&graph_next[l]);
    } else {
      break;
    }
  }

  double ratio = (50.0f + (double)tolerance) / (50.0f - (double)tolerance);
  double tol   = std::max(ratio, 1 - ratio) - 1;

  // forms a bi-partition
  partition(*graph_curr, tol);

  int levels = graphs.size();

  // Uncoarsening and Refine phase
  for (int level = levels - 1; level >= 0; level--) {

    if (level < levels - 1) {

      // initialize partitions of children nodes
      galois::do_all(galois::iterate(*graphs[level]), [&](GNode n) {
        auto parent = graphs[level]->getData(n).curr_comm_ass;
        graphs[level]->getData(n).setPart(
            graph_curr->getData(parent).getPart());
      });
    }
    graph_curr = graphs[level];

    extendBoundary(*graphs[level], 1);

    if (level < levels - 1)
      extendBoundary(*graphs[level], 0);

    fixLeafNodes(*graph_curr);
  }
  galois::gPrint("Phases : ", phase, "\n");
  galois::gPrint("Iter : ", iter, "\n");
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url, &inputFile);

  if (!symmetricGraph) {
    GALOIS_DIE("This application requires a symmetric graph input;"
               " please use the -symmetricGraph flag "
               " to indicate the input is a symmetric graph.");
  }

  galois::StatTimer totalTime("TimerTotal");
  totalTime.start();

  Graph graph;
  Graph graph_next;
  Graph* graph_curr;

  std::cout << "Reading from file: " << inputFile << "\n";
  std::cout << "[WARNING:] Make sure " << inputFile
            << " is symmetric graph without duplicate edges\n";
  galois::graphs::readGraph(graph, inputFile);
  std::cout << "Read " << graph.size() << " nodes, " << graph.sizeEdges()
            << " edges\n";

  graph_curr = &graph;

  /*
   *Initialize node cluster id.
   */
  galois::do_all(galois::iterate(*graph_curr),
                 [&](GNode n) { graph.getData(n).setWeight(1); });

  printGraphCharateristics(*graph_curr);

  for (auto n : *graph_curr) {

    for (auto edge : graph_curr->edges(n)) {
      graph_curr->getEdgeData(edge).weight = 1;
    }
  }

  galois::StatTimer execTime("Timer_0");
  execTime.start();
  runMultiLevelAlgorithm(*graph_curr, min_graph_size, c_threshold);
  execTime.stop();

  // output partitions
  if (output_CID) {
    printPartitions(graph, output_CID_filename);
  }

  // output edge cut
  uint64_t cut = computingCut(*graph_curr);
  std::cout << "edge cut: " << cut << std::endl;

  outputPartitionStats(*graph_curr);

  totalTime.stop();

  return 0;
}
