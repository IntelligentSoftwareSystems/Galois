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

#include "clustering.h"
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
#include <deque>
#include <type_traits>

static const char* name = "Louvain Clustering";

static const char* desc = "Cluster nodes of the graph using Louvain Clustering";

static const char* url = "louvain_clustering";

enum Algo { coloring, foreach, delay, doall };

static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<Algo> algo(
    "algo", cll::desc("Choose an algorithm:"),
    cll::values(clEnumValN(Algo::coloring, "Coloring",
                           "Using colors to mitigate conflicts"),
                clEnumValN(Algo::foreach, "Foreach",
                           "Using galois for_each for conflict mitigation"),
                clEnumValN(Algo::delay, "Delay",
                           "Using galois for_each for conflict mitigation but "
                           "delay the updation"),
                clEnumValN(Algo::doall, "Doall",
                           "Using galois for_each for conflict mitigation")),
    cll::init(Algo::foreach));

// Maintain community information
struct Comm {
  std::atomic<uint64_t> size;
  std::atomic<EdgeTy> degree_wt;
  EdgeTy internal_edge_wt;
};

typedef galois::LargeArray<Comm> CommArray;

// Graph Node information
struct Node {
  uint64_t prev_comm_ass;
  uint64_t curr_comm_ass;
  EdgeTy degree_wt;
  int64_t colorId;
};

using Graph = galois::graphs::LC_CSR_Graph<Node, EdgeTy>::with_no_lockable<
    false>::type::with_numa_alloc<true>::type;
using GNode = Graph::GraphNode;

double algoLouvainWithLocking(Graph& graph, double lower, double threshold,
                              uint32_t& iter) {
  galois::StatTimer TimerClusteringTotal("Timer_Clustering_Total");
  TimerClusteringTotal.start();

  galois::gPrint("Inside algoLouvainWithLocking\n");

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

  /* Initialization each node to its own cluster */
  galois::do_all(galois::iterate(graph), [&graph](GNode n) {
    graph.getData(n).curr_comm_ass = n;
    graph.getData(n).prev_comm_ass = n;
  });

  galois::gPrint("Init Done\n");
  /* Calculate the weighted degree sum for each vertex */
  sumVertexDegreeWeight(graph, c_info);
  galois::gPrint("c_info[0] : ", c_info[0].degree_wt.load(), "\n");

  /* Compute the total weight (2m) and 1/2m terms */
  constant_for_second_term = calConstantForSecondTerm(graph);
  galois::gPrint("constant_for_second_term : ", constant_for_second_term, "\n");

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
            // Find the max gain in modularity
            local_target =
                maxModularity(cluster_local_map, counter, self_loop_wt, c_info,
                              n_data.degree_wt, n_data.curr_comm_ass,
                              constant_for_second_term);

          } else {
            local_target = UNASSIGNED;
          }

          /* Update cluster info */
          if (local_target != n_data.curr_comm_ass &&
              local_target != UNASSIGNED) {

            galois::atomicAdd(c_info[local_target].degree_wt, n_data.degree_wt);
            galois::atomicAdd(c_info[local_target].size, (uint64_t)1);
            galois::atomicSubtract(c_info[n_data.curr_comm_ass].degree_wt,
                                   n_data.degree_wt);
            galois::atomicSubtract(c_info[n_data.curr_comm_ass].size,
                                   (uint64_t)1);

            /* Set the new cluster id */
            n_data.curr_comm_ass = local_target;
          }
        },
        galois::loopname("louvain algo: Phase 1"), galois::no_pushes());

    /* Calculate the overall modularity */
    double e_xx = 0;
    double a2_x = 0;

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

double algoLouvainWithoutLockingDoAll(Graph& graph, double lower,
                                      double threshold, uint32_t& iter) {

  galois::StatTimer TimerClusteringTotal("Timer_Clustering_Total");
  TimerClusteringTotal.start();

  galois::gPrint("Inside algoLouvainWithLocking\n");

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

  /* Initialization each node to its own cluster */
  galois::do_all(galois::iterate(graph), [&graph](GNode n) {
    graph.getData(n).curr_comm_ass = n;
    graph.getData(n).prev_comm_ass = n;
    graph.getData(n).colorId       = -1;
  });

  galois::gPrint("Init Done\n");
  /* Calculate the weighted degree sum for each vertex */
  sumVertexDegreeWeight(graph, c_info);
  galois::gPrint("c_info[0] : ", c_info[0].degree_wt.load(), "\n");

  /* Compute the total weight (2m) and 1/2m terms */
  constant_for_second_term = calConstantForSecondTerm(graph);
  galois::gPrint("constant_for_second_term : ", constant_for_second_term, "\n");

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
    });

    galois::do_all(
        galois::iterate(graph),
        [&](GNode n) {
          auto& n_data    = graph.getData(n, flag_write_lock);
          uint64_t degree = std::distance(graph.edge_begin(n, flag_no_lock),
                                          graph.edge_end(n, flag_no_lock));
          uint64_t local_target = UNASSIGNED;
          std::map<uint64_t, uint64_t>
              cluster_local_map; // Map each neighbor's cluster to local number:
                                 // Community --> Index
          std::vector<EdgeTy> counter; // Number of edges to each unique cluster
          EdgeTy self_loop_wt = 0;

          if (degree > 0) {
            findNeighboringClusters(graph, n, cluster_local_map, counter,
                                    self_loop_wt);
            // Find the max gain in modularity
            local_target = maxModularityWithoutSwaps(
                cluster_local_map, counter, self_loop_wt, c_info,
                n_data.degree_wt, n_data.curr_comm_ass,
                constant_for_second_term);

          } else {
            local_target = UNASSIGNED;
          }

          /* Update cluster info */
          if (local_target != n_data.curr_comm_ass &&
              local_target != UNASSIGNED) {

            galois::atomicAdd(c_info[local_target].degree_wt, n_data.degree_wt);
            galois::atomicAdd(c_info[local_target].size, (uint64_t)1);
            galois::atomicSubtract(c_info[n_data.curr_comm_ass].degree_wt,
                                   n_data.degree_wt);
            galois::atomicSubtract(c_info[n_data.curr_comm_ass].size,
                                   (uint64_t)1);

            /* Set the new cluster id */
            n_data.curr_comm_ass = local_target;
          }
        },
        galois::loopname("louvain algo: Phase 1"));

    /* Calculate the overall modularity */
    double e_xx = 0;
    double a2_x = 0;

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

double algoLouvainWithLockingDelayUpdate(Graph& graph, double lower,
                                         double threshold, uint32_t& iter) {
  galois::gPrint("Inside algoLouvainWithLockingDelay\n");

  galois::StatTimer TimerClusteringTotal("Timer_Clustering_Total");
  TimerClusteringTotal.start();

  CommArray c_info;   // Community info
  CommArray c_update; // Used for updating community

  /* Variables needed for Modularity calculation */
  double constant_for_second_term;
  double prev_mod      = -1; // lower;
  double curr_mod      = -1;
  double threshold_mod = threshold;
  uint32_t num_iter    = iter;

  /*** Initialization ***/
  c_info.allocateBlocked(graph.size());
  c_update.allocateBlocked(graph.size());

  /* Initialization each node to its own cluster */
  galois::do_all(galois::iterate(graph), [&graph](GNode n) {
    graph.getData(n).curr_comm_ass = n;
    graph.getData(n).prev_comm_ass = n;
    graph.getData(n).colorId       = -1;
  });

  galois::gPrint("Init Done\n");
  /* Calculate the weighted degree sum for each vertex */
  sumVertexDegreeWeight(graph, c_info);
  galois::gPrint("c_info[5] : ", c_info[0].degree_wt.load(), "\n");

  /* Compute the total weight (2m) and 1/2m terms */
  constant_for_second_term = calConstantForSecondTerm(graph);
  galois::gPrint("constant_for_second_term : ", constant_for_second_term, "\n");

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
    });

    std::vector<uint64_t> local_target(graph.size(), UNASSIGNED);
    galois::GAccumulator<uint32_t> syncRound;
    galois::do_all(
        galois::iterate(graph),
        [&](GNode n) {
          auto& n_data    = graph.getData(n, flag_write_lock);
          uint64_t degree = std::distance(graph.edge_begin(n, flag_no_lock),
                                          graph.edge_end(n, flag_no_lock));
          std::map<uint64_t, uint64_t>
              cluster_local_map; // Map each neighbor's cluster to local number:
                                 // Community --> Index
          std::vector<EdgeTy> counter; // Number of edges to each unique cluster
          EdgeTy self_loop_wt = 0;

          if (degree > 0) {
            findNeighboringClusters(graph, n, cluster_local_map, counter,
                                    self_loop_wt);
            // Find the max gain in modularity
            local_target[n] =
                maxModularity(cluster_local_map, counter, self_loop_wt, c_info,
                              n_data.degree_wt, n_data.curr_comm_ass,
                              constant_for_second_term);
          } else {
            local_target[n] = UNASSIGNED;
          }

          /* Update cluster info */
          if (local_target[n] != n_data.curr_comm_ass &&
              local_target[n] != UNASSIGNED) {

            galois::atomicAdd(c_update[local_target[n]].degree_wt,
                              n_data.degree_wt);
            galois::atomicAdd(c_update[local_target[n]].size, (uint64_t)1);
            galois::atomicSubtract(c_update[n_data.curr_comm_ass].degree_wt,
                                   n_data.degree_wt);
            galois::atomicSubtract(c_update[n_data.curr_comm_ass].size,
                                   (uint64_t)1);
          }
        },
        galois::loopname("louvain algo: Phase 1"));

    /* Calculate the overall modularity */
    double e_xx = 0;
    double a2_x = 0;
    curr_mod    = calModularityDelay(graph, c_info, c_update, e_xx, a2_x,
                                  constant_for_second_term, local_target);
    galois::gPrint(num_iter, "        ", e_xx, "        ", a2_x, "        ",
                   lower, "      ", prev_mod, "       ", curr_mod, "\n");

    if ((curr_mod - prev_mod) < threshold_mod) {
      galois::gPrint("Modularity gain: ", (curr_mod - prev_mod), " < ",
                     threshold_mod, " \n");
      prev_mod = curr_mod;
      break;
    }

    prev_mod = curr_mod;
    if (prev_mod < lower)
      prev_mod = lower;

    galois::do_all(galois::iterate(graph), [&](GNode n) {
      auto& n_data         = graph.getData(n, flag_no_lock);
      n_data.prev_comm_ass = n_data.curr_comm_ass;
      n_data.curr_comm_ass = local_target[n];
      galois::atomicAdd(c_info[n].size, c_update[n].size.load());
      galois::atomicAdd(c_info[n].degree_wt, c_update[n].degree_wt.load());

      c_update[n].size      = 0;
      c_update[n].degree_wt = 0;
    });

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

uint64_t coloringDistanceOne(Graph& graph) {
  galois::for_each(
      galois::iterate(graph),
      [&](GNode n, auto&) {
        auto& n_data = graph.getData(n, flag_write_lock);

        /* Grab lock on neighbours: Cautious operator */
        for (auto ii = graph.edge_begin(n, flag_write_lock);
             ii != graph.edge_end(n, flag_write_lock); ++ii) {
          graph.getData(graph.getEdgeDst(ii),
                        flag_write_lock); // TODO: Can we use read lock?
        }

        int64_t max_color = -1;
        int64_t my_color  = 0;
        int64_t degree = std::distance(graph.edge_begin(n), graph.edge_end(n));
        if (degree > 0) {
          std::vector<bool> isColorSet;
          isColorSet.resize(degree, false);
          for (auto ii = graph.edge_begin(n, flag_write_lock);
               ii != graph.edge_end(n, flag_write_lock); ++ii) {
            auto dst = graph.getEdgeDst(ii);
            if (dst == n)
              continue;

            auto& dst_data = graph.getData(
                dst, flag_write_lock); // TODO: Can we use read lock?
            if (dst_data.colorId >= 0) {
              if (dst_data.colorId >= degree)
                isColorSet.resize(dst_data.colorId);

              isColorSet[dst_data.colorId] = true;
              if ((dst_data.colorId > max_color)) {
                max_color = dst_data.colorId;
              }
            }
          }

          if (max_color >= 0) {
            /* Assign color */
            for (; my_color <= max_color; my_color++) {
              if (isColorSet[my_color] == false) {
                break;
              }
            }

            if (my_color == max_color)
              my_color++;
          }
        }
        n_data.colorId = my_color;
      },
      galois::loopname("Coloring loop"));

  galois::gPrint("Checking for conflicts\n");
  /* Check for conflicts */
  galois::GAccumulator<uint64_t> conflicts;
  galois::do_all(
      galois::iterate(graph),
      [&](GNode n) {
        auto& n_data = graph.getData(n, flag_no_lock);
        for (auto ii = graph.edge_begin(n, flag_write_lock);
             ii != graph.edge_end(n, flag_write_lock); ++ii) {
          auto dst       = graph.getEdgeDst(ii);
          auto& dst_data = graph.getData(dst, flag_no_lock);
          if (dst_data.colorId == n_data.colorId)
            conflicts += 1;
        }
      },
      galois::loopname("Coloring conflicts"));
  galois::gPrint("WARNING: Conflicts found : ", conflicts.reduce(), "\n");

  int64_t num_colors = 0;
  for (GNode n = 0; n < graph.size(); ++n) {
    int64_t color = graph.getData(n, flag_no_lock).colorId;
    if (color > num_colors)
      num_colors = color;
  }

  return num_colors;
}

double algoLouvainWithColoring(Graph& graph, double lower, double threshold,
                               uint32_t& iter) {

  galois::StatTimer TimerClusteringTotal("Timer_Clustering_Total");
  TimerClusteringTotal.start();

  galois::gPrint("Inside algoLouvainWithColoring\n");

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

  /* Initialization each node to its own cluster */
  galois::do_all(galois::iterate(graph), [&graph](GNode n) {
    graph.getData(n).curr_comm_ass = n;
    graph.getData(n).prev_comm_ass = n;
    graph.getData(n).colorId       = -1;
  });

  galois::gPrint("Coloring\n");
  galois::StatTimer TimerColoring("Timer_Cloring");
  TimerColoring.start();
  int64_t num_colors = coloringDistanceOne(graph);
  TimerColoring.stop();

  /* Calculate the weighted degree sum for each vertex */
  sumVertexDegreeWeight(graph, c_info);
  galois::gPrint("c_info[5] : ", c_info[0].degree_wt.load(), "\n");

  /* Compute the total weight (2m) and 1/2m terms */
  constant_for_second_term = calConstantForSecondTerm(graph);
  galois::gPrint("constant_for_second_term : ", constant_for_second_term, "\n");

  galois::gPrint("============================================================="
                 "===========================================\n");
  galois::gPrint(
      "Itr      Explore_xx            A_x2           Prev-Mod           "
      "Curr-Mod         Time-1(s)       Time-2(s)        T/Itr(s)\n");
  galois::gPrint("============================================================="
                 "===========================================\n");

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    c_update[n].degree_wt = 0;
    c_update[n].size      = 0;
  });

  galois::StatTimer TimerClusteringWhile("Timer_Clustering_While");
  TimerClusteringWhile.start();
  while (true) {
    num_iter++;

    for (int64_t c = 0; c < num_colors; ++c) {
      // galois::gPrint("Color : ", c, "\n");
      galois::do_all(
          galois::iterate(graph),
          [&](GNode n) {
            auto& n_data = graph.getData(n, flag_write_lock);
            if (n_data.colorId == c) {
              uint64_t degree = std::distance(graph.edge_begin(n, flag_no_lock),
                                              graph.edge_end(n, flag_no_lock));
              uint64_t local_target = UNASSIGNED;
              std::map<uint64_t, uint64_t>
                  cluster_local_map; // Map each neighbor's cluster to local
                                     // number: Community --> Index
              std::vector<EdgeTy>
                  counter; // Number of edges to each unique cluster
              EdgeTy self_loop_wt = 0;

              if (degree > 0) {
                findNeighboringClusters(graph, n, cluster_local_map, counter,
                                        self_loop_wt);
                // Find the max gain in modularity
                local_target = maxModularity(
                    cluster_local_map, counter, self_loop_wt, c_info,
                    n_data.degree_wt, n_data.curr_comm_ass,
                    constant_for_second_term);
              } else {
                local_target = UNASSIGNED;
              }
              /* Update cluster info */
              if (local_target != n_data.curr_comm_ass &&
                  local_target != UNASSIGNED) {
                galois::atomicAdd(c_update[local_target].degree_wt,
                                  n_data.degree_wt);
                galois::atomicAdd(c_update[local_target].size, (uint64_t)1);
                galois::atomicSubtract(c_update[n_data.curr_comm_ass].degree_wt,
                                       n_data.degree_wt);
                galois::atomicSubtract(c_update[n_data.curr_comm_ass].size,
                                       (uint64_t)1);
                /* Set the new cluster id */
                n_data.curr_comm_ass = local_target;
              }
            }
          },
          galois::loopname("louvain algo: Phase 1"));

      galois::do_all(galois::iterate(graph), [&](GNode n) {
        galois::atomicAdd(c_info[n].size, c_update[n].size.load());
        galois::atomicAdd(c_info[n].degree_wt, c_update[n].degree_wt.load());
        c_update[n].size      = 0;
        c_update[n].degree_wt = 0;
      });
    }

    /* Calculate the overall modularity */
    double e_xx = 0;
    double a2_x = 0;
    curr_mod =
        calModularity(graph, c_info, e_xx, a2_x, constant_for_second_term);

    galois::gPrint(num_iter, "        ", e_xx, "        ", a2_x, "        ",
                   prev_mod, "       ", curr_mod, "\n");

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

void runMultiPhaseLouvainAlgorithm(Graph& graph, uint32_t min_graph_size,
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
  while (true) {
    iter++;
    phase++;
    galois::gPrint("Starting Phase : ", phase, "\n");
    galois::gPrint("Graph size : ", (*graph_curr).size(), "\n");

    if ((*graph_curr).size() > min_graph_size) {

      switch (algo) {
      case coloring:
        curr_mod =
            algoLouvainWithColoring(*graph_curr, curr_mod, c_threshold, iter);
        break;
      case foreach:
        curr_mod =
            algoLouvainWithLocking(*graph_curr, curr_mod, c_threshold, iter);
        break;
      case doall:
        curr_mod = algoLouvainWithoutLockingDoAll(*graph_curr, curr_mod,
                                                  c_threshold, iter);
        break;
      case delay:
        curr_mod = algoLouvainWithLockingDelayUpdate(*graph_curr, curr_mod,
                                                     c_threshold, iter);
        break;
      default:
        std::abort();
      }
    }

    uint64_t num_unique_clusters = renumberClustersContiguously(*graph_curr);
    galois::gPrint(
        "Number of unique clusters (renumber): ", num_unique_clusters, "\n");

    galois::gPrint("Prev_mod main: ", prev_mod, "\n");
    if (iter < max_iter && (curr_mod - prev_mod) > threshold) {
      if (!enable_VF && phase == 1) {
        assert(num_nodes_orig == (*graph_curr).size());
        galois::do_all(galois::iterate(*graph_curr), [&](GNode n) {
          clusters_orig[n] =
              (*graph_curr).getData(n, flag_no_lock).curr_comm_ass;
        });
      } else {
        galois::do_all(
            galois::iterate((uint64_t)0, num_nodes_orig), [&](GNode n) {
              if (clusters_orig[n] != UNASSIGNED) {
                assert(clusters_orig[n] < graph_curr->size());
                clusters_orig[n] = (*graph_curr)
                                       .getData(clusters_orig[n], flag_no_lock)
                                       .curr_comm_ass;
              }
            });
      }
      buildNextLevelGraph(*graph_curr, graph_next, num_unique_clusters);
      prev_mod   = curr_mod;
      graph_curr = &graph_next;
      printGraphCharateristics(*graph_curr);
    } else {
      break;
    }
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

    // Build new graph to remove the isolated nodes
    buildNextLevelGraph(*graph_curr, graph_next, num_unique_clusters);
    graph_curr = &graph_next;
    printGraphCharateristics(*graph_curr);
  } else {

    /*
     *Initialize node cluster id.
     */
    galois::do_all(galois::iterate(*graph_curr),
                   [&](GNode n) { clusters_orig[n] = -1; });

    printGraphCharateristics(*graph_curr);
  }

  galois::StatTimer execTime("Timer_0");
  execTime.start();
  runMultiPhaseLouvainAlgorithm(*graph_curr, min_graph_size, c_threshold,
                                clusters_orig);
  execTime.stop();

  /*
   * Sanity check: Check modularity at the end
   */
  checkModularity<Graph, CommArray>(graph, clusters_orig);
  if (output_CID) {
    printNodeClusterId(graph, output_CID_filename);
  }

  totalTime.stop();

  return 0;
}
