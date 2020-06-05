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

#ifndef CLUSTERING_H
#define CLUSTERING_H

#include "galois/Galois.h"
#include "galois/AtomicHelpers.h"
#include "galois/LargeArray.h"

#include "llvm/Support/CommandLine.h"

#include <random>
#include <fstream>

namespace cll = llvm::cl;
static cll::opt<bool>
    enable_VF("enable_VF",
              cll::desc("Flag to enable vertex following optimization."),
              cll::init(false));

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

static cll::opt<double>
    resolution("resolution", cll::desc("Resolution for CPM quality function."),
               cll::init(1.0));

static cll::opt<double>
    randomness("randomness",
               cll::desc("Randomness factor for refining clusters in Leiden."),
               cll::init(0.01));

static cll::opt<uint32_t>
    min_graph_size("min_graph_size", cll::desc("Minimum coarsened graph size"),
                   cll::init(100));

/*
 * Typedefs
 */
constexpr static const uint64_t INF_VAL =
    std::numeric_limits<uint64_t>::max() / 2 - 1;
constexpr static const uint64_t UNASSIGNED =
    std::numeric_limits<uint64_t>::max();
constexpr static const double DOUBLE_MAX =
    std::numeric_limits<double>::max() / 4;

constexpr galois::MethodFlag flag_no_lock    = galois::MethodFlag::UNPROTECTED;
constexpr galois::MethodFlag flag_read_lock  = galois::MethodFlag::READ;
constexpr galois::MethodFlag flag_write_lock = galois::MethodFlag::WRITE;

typedef galois::LargeArray<uint64_t> largeArray;
typedef float EdgeTy;
// typedef uint32_t EdgeTy;
typedef galois::LargeArray<EdgeTy> largeArrayEdgeTy;

template <typename GraphTy>
void printGraphCharateristics(GraphTy& graph) {

  galois::gPrint("/******************************************/\n");
  galois::gPrint("/************ Graph Properties ************/\n");
  galois::gPrint("/******************************************/\n");
  galois::gPrint("Number of Nodes: ", graph.size(), "\n");
  galois::gPrint("Number of Edges: ", graph.sizeEdges(), "\n");
}

/**
 * Algorithm to find the best cluster for the node
 * to move to among its neighbors.
 */
template <typename GraphTy>
void findNeighboringClusters(GraphTy& graph, typename GraphTy::GraphNode& n,
                             std::map<uint64_t, uint64_t>& cluster_local_map,
                             std::vector<EdgeTy>& counter,
                             EdgeTy& self_loop_wt) {
  using GNode = typename GraphTy::GraphNode;
  for (auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
    graph.getData(graph.getEdgeDst(ii), flag_write_lock);
  }

  uint64_t num_unique_clusters = 0;
  /**
   * Add the node's current cluster to be considered
   * for movement as well
   */
  cluster_local_map[graph.getData(n).curr_comm_ass] =
      0;                // Add n's current cluster
  counter.push_back(0); // Initialize the counter to zero (no edges incident
                        // yet)
  num_unique_clusters++;

  // Assuming we have grabbed lock on all the neighbors
  for (auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
    GNode dst = graph.getEdgeDst(ii);
    auto edge_wt =
        graph.getEdgeData(ii, flag_no_lock); // Self loop weights is recorded
    if (dst == n) {
      self_loop_wt += edge_wt; // Self loop weights is recorded
    }
    auto stored_already = cluster_local_map.find(
        graph.getData(dst).curr_comm_ass); // Check if it already exists
    if (stored_already != cluster_local_map.end()) {
      counter[stored_already->second] += edge_wt;
    } else {
      cluster_local_map[graph.getData(dst).curr_comm_ass] = num_unique_clusters;
      counter.push_back(edge_wt);
      num_unique_clusters++;
    }
  } // End edge loop
  return;
}
template <typename GraphTy>
uint64_t vertexFollowing(GraphTy& graph) {
  using GNode = typename GraphTy::GraphNode;
  // Initialize each node to its own cluster
  galois::do_all(galois::iterate(graph),
                 [&graph](GNode n) { graph.getData(n).curr_comm_ass = n; });

  // Remove isolated and degree-one nodes
  galois::GAccumulator<uint64_t> isolatedNodes;
  galois::do_all(galois::iterate(graph), [&](GNode n) {
    auto& n_data = graph.getData(n);
    uint64_t degree =
        std::distance(graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
                      graph.edge_end(n, galois::MethodFlag::UNPROTECTED));
    if (degree == 0) {
      isolatedNodes += 1;
      n_data.curr_comm_ass = UNASSIGNED;
    } else {
      if (degree == 1) {
        // Check if the destination has degree greater than one
        auto dst = graph.getEdgeDst(
            graph.edge_end(n, galois::MethodFlag::UNPROTECTED));
        uint64_t dst_degree = std::distance(
            graph.edge_begin(dst, galois::MethodFlag::UNPROTECTED),
            graph.edge_end(dst, galois::MethodFlag::UNPROTECTED));
        if ((dst_degree > 1 || (n > dst))) {
          isolatedNodes += 1;
          n_data.curr_comm_ass = graph.getData(dst).curr_comm_ass;
        }
      }
    }
  });
  // The number of isolated nodes that can be removed
  return isolatedNodes.reduce();
}

template <typename GraphTy, typename CommArrayTy>
void sumVertexDegreeWeight(GraphTy& graph, CommArrayTy& c_info) {
  using GNode = typename GraphTy::GraphNode;
  galois::do_all(galois::iterate(graph), [&](GNode n) {
    EdgeTy total_weight = 0;
    auto& n_data        = graph.getData(n);
    for (auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
      total_weight += graph.getEdgeData(ii, flag_no_lock);
    }
    n_data.degree_wt    = total_weight;
    c_info[n].degree_wt = total_weight;
    c_info[n].size      = 1;
  });
}
template <typename GraphTy, typename CommArrayTy>
void sumVertexDegreeWeightWithNodeWeight(GraphTy& graph, CommArrayTy& c_info) {
  using GNode = typename GraphTy::GraphNode;
  galois::do_all(galois::iterate(graph), [&](GNode n) {
    EdgeTy total_weight = 0;
    auto& n_data        = graph.getData(n);
    for (auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
      total_weight += graph.getEdgeData(ii, flag_no_lock);
    }
    n_data.degree_wt    = total_weight;
    c_info[n].degree_wt = total_weight;
    c_info[n].size      = 1;
    c_info[n].node_wt.store(n_data.node_wt);
  });
}

template <typename GraphTy, typename CommArrayTy>
void sumClusterWeight(GraphTy& graph, CommArrayTy& c_info) {
  using GNode = typename GraphTy::GraphNode;
  galois::do_all(galois::iterate(graph), [&](GNode n) {
    EdgeTy total_weight = 0;
    auto& n_data        = graph.getData(n);
    for (auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
      total_weight += graph.getEdgeData(ii, flag_no_lock);
    }
    n_data.degree_wt    = total_weight;
    c_info[n].degree_wt = 0;
  });

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    auto& n_data = graph.getData(n);
    if (n_data.curr_comm_ass != UNASSIGNED)
      galois::atomicAdd(c_info[n_data.curr_comm_ass].degree_wt,
                        n_data.degree_wt);
  });
}

template <typename GraphTy>
double calConstantForSecondTerm(GraphTy& graph) {
  using GNode = typename GraphTy::GraphNode;
  /**
   * Using double to avoid overflow
   */
  galois::GAccumulator<double> local_weight;
  galois::do_all(galois::iterate(graph), [&graph, &local_weight](GNode n) {
    local_weight += graph.getData(n).degree_wt;
  });
  /* This is twice since graph is symmetric */
  double total_edge_weight_twice = local_weight.reduce();
  return 1 / total_edge_weight_twice;
}

template <typename GraphTy, typename CommArrayTy>
uint64_t maxCPMQuality(std::map<uint64_t, uint64_t>& cluster_local_map,
                       std::vector<EdgeTy>& counter, EdgeTy self_loop_wt,
                       CommArrayTy& c_info, uint64_t node_wt, uint64_t sc) {

  uint64_t max_index = sc; // Assign the initial value as self community
  double cur_gain    = 0;
  double max_gain    = 0;
  double eix         = counter[0] - self_loop_wt;
  double eiy         = 0;
  double size_x      = (double)(c_info[sc].node_wt - node_wt);
  double size_y      = 0;

  auto stored_already = cluster_local_map.begin();
  do {
    if (sc != stored_already->first) {

      eiy =
          counter[stored_already->second]; // Total edges incident on cluster y
      size_y = c_info[stored_already->first].node_wt;

      cur_gain = 2.0f * (double)(eiy - eix) -
                 resolution * node_wt * (double)(size_y - size_x);
      if ((cur_gain > max_gain) || ((cur_gain == max_gain) && (cur_gain != 0) &&
                                    (stored_already->first < max_index))) {
        max_gain  = cur_gain;
        max_index = stored_already->first;
      }
    }
    stored_already++; // Explore next cluster
  } while (stored_already != cluster_local_map.end());

  if ((c_info[max_index].size == 1 && c_info[sc].size == 1 && max_index > sc)) {
    max_index = sc;
  }
  assert(max_gain >= 0);
  return max_index;
}

template <typename GraphTy, typename CommArrayTy>
double calCPMQuality(GraphTy& graph, CommArrayTy& c_info, double& e_xx,
                     double& a2_x, double& constant_for_second_term) {

  using GNode = typename GraphTy::GraphNode;
  /* Variables needed for Modularity calculation */
  double mod = -1;

  std::cout << "graph size: " << graph.size() << "\n";
  largeArrayEdgeTy cluster_wt_internal;

  /*** Initialization ***/
  cluster_wt_internal.allocateBlocked(graph.size());

  /* Calculate the overall modularity */
  galois::GAccumulator<double> acc_e_xx;
  galois::GAccumulator<double> acc_a2_x;

  galois::do_all(galois::iterate(graph),
                 [&](GNode n) { cluster_wt_internal[n] = 0; });

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    auto n_data = graph.getData(n);
    for (auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
      if (graph.getData(graph.getEdgeDst(ii)).curr_comm_ass ==
          n_data.curr_comm_ass) {
        cluster_wt_internal[n] += graph.getEdgeData(ii);
      }
    }
  });

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    acc_e_xx += cluster_wt_internal[n];
    acc_a2_x +=
        (double)(c_info[n].node_wt) * ((double)(c_info[n].node_wt - 1) * 0.5f);
    // acc_a2_x += (double) (c_info[n].node_wt) * ((double) (c_info[n].node_wt)
    // * resolution);
  });

  e_xx = acc_e_xx.reduce();
  a2_x = acc_a2_x.reduce();
  mod  = (e_xx - a2_x) * (double)constant_for_second_term;

  return mod;
}

template <typename CommArrayTy>
uint64_t maxModularity(std::map<uint64_t, uint64_t>& cluster_local_map,
                       std::vector<EdgeTy>& counter, EdgeTy self_loop_wt,
                       CommArrayTy& c_info, EdgeTy degree_wt, uint64_t sc,
                       double constant) {

  uint64_t max_index = sc; // Assign the intial value as self community
  double cur_gain    = 0;
  double max_gain    = 0;
  double eix         = counter[0] - self_loop_wt;
  double ax          = c_info[sc].degree_wt - degree_wt;
  double eiy         = 0;
  double ay          = 0;

  auto stored_already = cluster_local_map.begin();
  do {
    if (sc != stored_already->first) {
      ay = c_info[stored_already->first].degree_wt; // Degree wt of cluster y
      eiy =
          counter[stored_already->second]; // Total edges incident on cluster y
      cur_gain = 2 * constant * (eiy - eix) +
                 2 * degree_wt * ((ax - ay) * constant * constant);

      if ((cur_gain > max_gain) || ((cur_gain == max_gain) && (cur_gain != 0) &&
                                    (stored_already->first < max_index))) {
        max_gain  = cur_gain;
        max_index = stored_already->first;
      }
    }
    stored_already++; // Explore next cluster
  } while (stored_already != cluster_local_map.end());

  if ((c_info[max_index].size == 1 && c_info[sc].size == 1 && max_index > sc)) {
    max_index = sc;
  }

  assert(max_gain >= 0);
  return max_index;
}

template <typename CommArrayTy>
uint64_t
maxModularityWithoutSwaps(std::map<uint64_t, uint64_t>& cluster_local_map,
                          std::vector<EdgeTy>& counter, uint64_t self_loop_wt,
                          CommArrayTy& c_info, EdgeTy degree_wt, uint64_t sc,
                          double constant) {

  uint64_t max_index = sc; // Assign the intial value as self community
  double cur_gain    = 0;
  double max_gain    = 0;
  double eix         = counter[0] - self_loop_wt;
  double ax          = c_info[sc].degree_wt - degree_wt;
  double eiy         = 0;
  double ay          = 0;

  auto stored_already = cluster_local_map.begin();
  do {
    if (sc != stored_already->first) {
      ay = c_info[stored_already->first].degree_wt; // Degree wt of cluster y

      if (ay < (ax + degree_wt)) {
        stored_already++;
        continue;
      } else if (ay == (ax + degree_wt) && stored_already->first > sc) {
        stored_already++;
        continue;
      }

      eiy =
          counter[stored_already->second]; // Total edges incident on cluster y
      cur_gain = 2 * constant * (eiy - eix) +
                 2 * degree_wt * ((ax - ay) * constant * constant);

      if ((cur_gain > max_gain) || ((cur_gain == max_gain) && (cur_gain != 0) &&
                                    (stored_already->first < max_index))) {
        max_gain  = cur_gain;
        max_index = stored_already->first;
      }
    }
    stored_already++; // Explore next cluster
  } while (stored_already != cluster_local_map.end());

  if ((c_info[max_index].size == 1 && c_info[sc].size == 1 && max_index > sc)) {
    max_index = sc;
  }

  assert(max_gain >= 0);
  return max_index;
}

template <typename GraphTy, typename CommArrayTy>
double calModularityDelay(GraphTy& graph, CommArrayTy& c_info,
                          CommArrayTy& c_update, double& e_xx, double& a2_x,
                          double& constant_for_second_term,
                          std::vector<uint64_t>& local_target) {
  using GNode = typename GraphTy::GraphNode;
  /* Variables needed for Modularity calculation */
  double mod = -1;

  largeArrayEdgeTy cluster_wt_internal;

  /*** Initialization ***/
  cluster_wt_internal.allocateBlocked(graph.size());

  /* Calculate the overall modularity */
  galois::GAccumulator<double> acc_e_xx;
  galois::GAccumulator<double> acc_a2_x;

  galois::do_all(galois::iterate(graph),
                 [&](GNode n) { cluster_wt_internal[n] = 0; });

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    for (auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
      if (local_target[graph.getEdgeDst(ii)] == local_target[n]) {
        cluster_wt_internal[n] += graph.getEdgeData(ii);
      }
    }
  });

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    acc_e_xx += cluster_wt_internal[n];
    acc_a2_x += (double)(c_info[n].degree_wt + c_update[n].degree_wt) *
                ((double)(c_info[n].degree_wt + c_update[n].degree_wt) *
                 (double)constant_for_second_term);
  });

  e_xx = acc_e_xx.reduce();
  a2_x = acc_a2_x.reduce();

  mod = e_xx * (double)constant_for_second_term -
        a2_x * (double)constant_for_second_term;
  return mod;
}

template <typename GraphTy, typename CommArrayTy>
double calModularity(GraphTy& graph, CommArrayTy& c_info, double& e_xx,
                     double& a2_x, double& constant_for_second_term) {
  using GNode = typename GraphTy::GraphNode;
  /* Variables needed for Modularity calculation */
  double mod = -1;

  largeArrayEdgeTy cluster_wt_internal;

  /*** Initialization ***/
  cluster_wt_internal.allocateBlocked(graph.size());

  /* Calculate the overall modularity */
  galois::GAccumulator<double> acc_e_xx;
  galois::GAccumulator<double> acc_a2_x;

  galois::do_all(galois::iterate(graph),
                 [&](GNode n) { cluster_wt_internal[n] = 0; });

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    auto n_data = graph.getData(n);
    for (auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
      if (graph.getData(graph.getEdgeDst(ii)).curr_comm_ass ==
          n_data.curr_comm_ass) {
        cluster_wt_internal[n] += graph.getEdgeData(ii);
      }
    }
  });

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    acc_e_xx += cluster_wt_internal[n];
    acc_a2_x +=
        (double)(c_info[n].degree_wt) *
        ((double)(c_info[n].degree_wt) * (double)constant_for_second_term);
  });

  e_xx = acc_e_xx.reduce();
  a2_x = acc_a2_x.reduce();

  mod = e_xx * (double)constant_for_second_term -
        a2_x * (double)constant_for_second_term;
  return mod;
}

/*
 * To compute the final modularity using prev cluster
 * assignments.
 */
template <typename GraphTy, typename CommArrayTy>
double calModularityFinal(GraphTy& graph) {
  using GNode     = typename GraphTy::GraphNode;
  using CommArray = CommArrayTy;

  CommArray c_info;   // Community info
  CommArray c_update; // Used for updating community

  /* Variables needed for Modularity calculation */
  double constant_for_second_term;
  double mod = -1;

  largeArrayEdgeTy cluster_wt_internal;

  /*** Initialization ***/
  c_info.allocateBlocked(graph.size());
  c_update.allocateBlocked(graph.size());
  cluster_wt_internal.allocateBlocked(graph.size());

  /* Calculate the weighted degree sum for each vertex */
  sumClusterWeight(graph, c_info);

  /* Compute the total weight (2m) and 1/2m terms */
  constant_for_second_term = calConstantForSecondTerm(graph);

  /* Calculate the overall modularity */
  double e_xx = 0;
  galois::GAccumulator<double> acc_e_xx;
  double a2_x = 0;
  galois::GAccumulator<double> acc_a2_x;

  galois::do_all(galois::iterate(graph),
                 [&](GNode n) { cluster_wt_internal[n] = 0; });

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    auto n_data = graph.getData(n);
    for (auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
      if (graph.getData(graph.getEdgeDst(ii)).curr_comm_ass ==
          n_data.curr_comm_ass) {
        // if(graph.getData(graph.getEdgeDst(ii)).prev_comm_ass ==
        // n_data.prev_comm_ass) {
        cluster_wt_internal[n] += graph.getEdgeData(ii);
      }
    }
  });

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    acc_e_xx += cluster_wt_internal[n];
    acc_a2_x +=
        (double)(c_info[n].degree_wt) *
        ((double)(c_info[n].degree_wt) * (double)constant_for_second_term);
  });

  e_xx = acc_e_xx.reduce();
  a2_x = acc_a2_x.reduce();

  mod = e_xx * (double)constant_for_second_term -
        a2_x * (double)constant_for_second_term;
  return mod;
}
template <typename GraphTy>
uint64_t renumberClustersContiguously(GraphTy& graph) {
  using GNode = typename GraphTy::GraphNode;
  std::map<uint64_t, uint64_t> cluster_local_map;
  uint64_t num_unique_clusters = 0;

  for (GNode n = 0; n < graph.size(); ++n) {
    auto& n_data = graph.getData(n, flag_no_lock);
    if (n_data.curr_comm_ass != UNASSIGNED) {
      assert(n_data.curr_comm_ass < graph.size());
      auto stored_already = cluster_local_map.find(n_data.curr_comm_ass);
      if (stored_already != cluster_local_map.end()) {
        n_data.curr_comm_ass = stored_already->second;
      } else {
        cluster_local_map[n_data.curr_comm_ass] = num_unique_clusters;
        n_data.curr_comm_ass                    = num_unique_clusters;
        num_unique_clusters++;
      }
    }
  }
  return num_unique_clusters;
}

template <typename GraphTy>
uint64_t renumberClustersContiguouslySubcomm(GraphTy& graph) {

  using GNode = typename GraphTy::GraphNode;
  std::map<uint64_t, uint64_t> cluster_local_map;
  uint64_t num_unique_clusters = 0;

  for (GNode n = 0; n < graph.size(); ++n) {
    auto& n_data = graph.getData(n, flag_no_lock);
    assert(n_data.curr_subcomm_ass != UNASSIGNED);
    assert(n_data.curr_subcomm_ass < graph.size());
    auto stored_already = cluster_local_map.find(n_data.curr_subcomm_ass);
    if (stored_already != cluster_local_map.end()) {
      n_data.curr_subcomm_ass = stored_already->second;
    } else {
      cluster_local_map[n_data.curr_subcomm_ass] = num_unique_clusters;
      n_data.curr_subcomm_ass                    = num_unique_clusters;
      num_unique_clusters++;
    }
  }

  return num_unique_clusters;
}

template <typename GraphTy>
uint64_t renumberClustersContiguouslyArray(largeArray& arr) {
  using GNode = typename GraphTy::GraphNode;
  std::map<uint64_t, uint64_t> cluster_local_map;
  uint64_t num_unique_clusters = 0;

  for (GNode n = 0; n < arr.size(); ++n) {
    if (arr[n] != UNASSIGNED) {
      assert(arr[n] < arr.size());
      auto stored_already = cluster_local_map.find(arr[n]);
      if (stored_already != cluster_local_map.end()) {
        arr[n] = stored_already->second;
      } else {
        cluster_local_map[arr[n]] = num_unique_clusters;
        arr[n]                    = num_unique_clusters;
        num_unique_clusters++;
      }
    }
  }
  return num_unique_clusters;
}

template <typename GraphTy>
void printGraph(GraphTy& graph) {
  using GNode = typename GraphTy::GraphNode;
  for (GNode n = 0; n < graph.size(); ++n) {
    for (auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
      galois::gPrint(n, " --> ", graph.getEdgeDst(ii), " , ",
                     graph.getEdgeData(ii), "\n");
    }
  }
}

template <typename GraphTy>
void printNodeClusterId(GraphTy& graph, std::string output_CID_filename) {
  using GNode = typename GraphTy::GraphNode;
  std::ofstream outputFile(output_CID_filename, std::ofstream::out);
  for (GNode n = 0; n < graph.size(); ++n) {
    outputFile << n << "  " << graph.getData(n).curr_comm_ass << "\n";
    // outputFile << graph.getData(n).curr_comm_ass << "\n";
  }
}

template <typename GraphTy, typename CommArrayTy>
void checkModularity(GraphTy& graph, largeArray& clusters_orig) {
  using GNode = typename GraphTy::GraphNode;
  galois::gPrint("checkModularity\n");

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    graph.getData(n, flag_no_lock).curr_comm_ass = clusters_orig[n];
  });

  uint64_t num_unique_clusters = renumberClustersContiguously(graph);
  galois::gPrint("Number of unique clusters (renumber): ", num_unique_clusters,
                 "\n");
  auto mod = calModularityFinal<GraphTy, CommArrayTy>(graph);
  galois::gPrint("FINAL MOD: ", mod, "\n");
}

/***********************************************
 ********** Leiden Routines ********************
 **********************************************/
uint64_t generateRandonNumber(uint64_t min, uint64_t max) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist6(
      min, max); // distribution in range [min, max]
  return dist6(rng);
}

uint64_t generateRandonNumberDouble(double min, double max) {
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(min,
                                       max); // distribution in range [min, max]
  return dis(gen);
}

template <typename CommArrayTy>
double diffCPMQuality(uint64_t curr_subcomm, uint64_t candidate_subcomm,
                      std::map<uint64_t, uint64_t>& cluster_local_map,
                      std::vector<EdgeTy>& counter, CommArrayTy& subcomm_info,
                      EdgeTy self_loop_wt) {

  uint64_t size_x = subcomm_info[curr_subcomm].node_wt;
  uint64_t size_y = subcomm_info[candidate_subcomm].node_wt;

  double diff =
      (double)(counter[cluster_local_map[candidate_subcomm]] -
               counter[cluster_local_map[curr_subcomm]] + self_loop_wt) +
      resolution * 0.5f *
          (double)((size_x * (size_x - 1) + size_y * (size_y - 1)) -
                   ((size_x - 1) * (size_x - 2) + size_y * (size_y + 1)));

  return diff;
}

template <typename GraphTy, typename CommArrayTy>
uint64_t getRandomSubcommunity(GraphTy& graph, uint64_t n,
                               CommArrayTy& subcomm_info,
                               uint64_t total_degree_wt,
                               double constant_for_second_term) {
  using GNode           = typename GraphTy::GraphNode;
  uint64_t curr_subcomm = graph.getData(n).curr_subcomm_ass;

  std::map<uint64_t, uint64_t>
      cluster_local_map; // Map each neighbor's subcommunity to local number:
                         // Subcommunity --> Index
  std::vector<EdgeTy> counter; // Number of edges to each unique subcommunity
  uint64_t num_unique_clusters = 1;

  cluster_local_map[curr_subcomm] = 0; // Add n's current subcommunity
  counter.push_back(0); // Initialize the counter to zero (no edges incident
                        // yet)

  EdgeTy self_loop_wt = 0;

  for (auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
    GNode dst = graph.getEdgeDst(ii);
    EdgeTy edge_wt =
        graph.getEdgeData(ii, flag_no_lock); // Self loop weights is recorded

    if (dst == n) {
      self_loop_wt += edge_wt; // Self loop weights is recorded
    }
    auto stored_already = cluster_local_map.find(
        graph.getData(dst).curr_subcomm_ass); // Check if it already exists
    if (stored_already != cluster_local_map.end()) {
      counter[stored_already->second] += edge_wt;
    } else {
      cluster_local_map[graph.getData(dst).curr_subcomm_ass] =
          num_unique_clusters;
      counter.push_back(edge_wt);
      num_unique_clusters++;
    }
  } // End edge loop

  std::map<uint64_t, uint64_t> new_cluster_local_map;
  std::vector<EdgeTy> new_counter;
  num_unique_clusters = 0;
  EdgeTy total        = 0;

  for (auto pair : cluster_local_map) {
    auto subcomm = pair.first;
    if (curr_subcomm == subcomm)
      continue;
    uint64_t subcomm_degree_wt = subcomm_info[subcomm].degree_wt;

    // check if subcommunity is well connected
    if (subcomm_info[subcomm].internal_edge_wt <
        constant_for_second_term * (double)subcomm_degree_wt *
            ((double)total_degree_wt - (double)subcomm_degree_wt))
      continue;
    if (diffCPMQuality(curr_subcomm, subcomm, cluster_local_map, counter,
                       subcomm_info, self_loop_wt) > 0) {
      new_cluster_local_map[subcomm] = num_unique_clusters;
      EdgeTy count                   = counter[cluster_local_map[subcomm]];
      new_counter.push_back(count);
      total += count;
    }
  }

  // Pick max community size
  uint64_t rand_idx = 1; // getRandomInt(0,total-1);

  uint64_t idx = 0;
  for (auto pair : new_cluster_local_map) {
    if (new_counter[idx] > rand_idx)
      return pair.first;
    rand_idx = rand_idx - new_counter[idx];
    idx++;
  }

  return UNASSIGNED;
}

template <typename GraphTy, typename CommTy>
uint64_t getRandomSubcommunity2(GraphTy& graph, typename GraphTy::GraphNode n,
                                CommTy& subcomm_info, uint64_t total_degree_wt,
                                uint64_t comm_id,
                                double constant_for_second_term) {
  using GNode  = typename GraphTy::GraphNode;
  auto& n_data = graph.getData(n);
  /*
   * Remove the currently selected node from its current cluster.
   * This causes the cluster to be empty.
   */
  subcomm_info[n_data.curr_subcomm_ass].node_wt          = 0;
  subcomm_info[n_data.curr_subcomm_ass].internal_edge_wt = 0;

  /*
   * Map each neighbor's subcommunity to local number: Subcommunity --> Index
   */
  std::map<uint64_t, uint64_t> cluster_local_map;

  /*
   * Edges weight to each unique subcommunity
   */
  std::vector<EdgeTy> counter;
  std::vector<uint64_t> neighboring_cluster_ids;

  /*
   * Identify the neighboring clusters of the currently selected
   * node, that is, the clusters with which the currently
   * selected node is connected. The old cluster of the currently
   * selected node is also included in the set of neighboring
   * clusters. In this way, it is always possible that the
   * currently selected node will be moved back to its old
   * cluster.
   */
  cluster_local_map[n_data.curr_subcomm_ass] = 0; // Add n's current
                                                  // subcommunity
  counter.push_back(0); // Initialize the counter to zero (no edges incident
                        // yet)
  neighboring_cluster_ids.push_back(n_data.curr_subcomm_ass);
  uint64_t num_unique_clusters = 1;

  EdgeTy self_loop_wt = 0;

  for (auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
    GNode dst = graph.getEdgeDst(ii);
    EdgeTy edge_wt =
        graph.getEdgeData(ii, flag_no_lock); // Self loop weights is recorded
    if (graph.getData(dst).curr_comm_ass == comm_id) {
      if (dst == n) {
        self_loop_wt += edge_wt; // Self loop weights is recorded
      }
      auto stored_already = cluster_local_map.find(
          graph.getData(dst).curr_subcomm_ass); // Check if it already exists
      if (stored_already != cluster_local_map.end()) {
        counter[stored_already->second] += edge_wt;
      } else {
        cluster_local_map[graph.getData(dst).curr_subcomm_ass] =
            num_unique_clusters;
        counter.push_back(edge_wt);
        neighboring_cluster_ids.push_back(graph.getData(dst).curr_subcomm_ass);
        num_unique_clusters++;
      }
    }
  } // End edge loop

  uint64_t best_cluster                            = n_data.curr_subcomm_ass;
  double max_quality_value_increment               = 0;
  double total_transformed_quality_value_increment = 0;
  double quality_value_increment                   = 0;
  std::vector<double> cum_transformed_quality_value_increment_per_cluster(
      num_unique_clusters);
  for (auto pair : cluster_local_map) {
    auto subcomm = pair.first;
    if (n_data.curr_subcomm_ass == subcomm)
      continue;

    uint64_t subcomm_node_wt   = subcomm_info[subcomm].node_wt;
    uint64_t subcomm_degree_wt = subcomm_info[subcomm].degree_wt;

    // check if subcommunity is well connected
    if (subcomm_info[subcomm].internal_edge_wt >=
        constant_for_second_term * (double)subcomm_degree_wt *
            ((double)total_degree_wt - (double)subcomm_degree_wt)) {

      quality_value_increment =
          counter[pair.second] - n_data.node_wt * subcomm_node_wt * resolution;

      if (quality_value_increment > max_quality_value_increment) {
        best_cluster                = subcomm;
        max_quality_value_increment = quality_value_increment;
      }

      if (quality_value_increment >= 0)
        total_transformed_quality_value_increment +=
            std::exp(quality_value_increment / randomness);
    }
    cum_transformed_quality_value_increment_per_cluster[pair.second] =
        total_transformed_quality_value_increment;
    counter[pair.second] = 0;
  }

  /*
   * Determine the neighboring cluster to which the currently
   * selected node will be moved.
   */
  int64_t min_idx, max_idx, mid_idx;
  uint64_t chosen_cluster;
  double r;
  if (total_transformed_quality_value_increment < DOUBLE_MAX) {
    r = total_transformed_quality_value_increment *
        generateRandonNumberDouble(0.0, 1.0);
    min_idx = -1;
    max_idx = num_unique_clusters + 1;
    while (min_idx < max_idx - 1) {
      mid_idx = (min_idx + max_idx) / 2;
      if (cum_transformed_quality_value_increment_per_cluster[mid_idx] >= r)
        max_idx = mid_idx;
      else
        min_idx = mid_idx;
    }
    chosen_cluster = neighboring_cluster_ids[max_idx];
  } else {
    chosen_cluster = best_cluster;
  }
  return chosen_cluster;
}
/**
 * Finds a clustering of the nodes in a network using the local merging
 * algorithm.
 *
 * <p>
 * The local merging algorithm starts from a singleton partition. It
 * performs a single iteration over the nodes in a network. Each node
 * belonging to a singleton cluster is considered for merging with another
 * cluster. This cluster is chosen randomly from all clusters that do not
 * result in a decrease in the quality function. The larger the increase in
 * the quality function, the more likely a cluster is to be chosen. The
 * strength of this effect is determined by the randomness parameter. The
 * higher the value of the randomness parameter, the stronger the
 * randomness in the choice of a cluster. The lower the value of the
 * randomness parameter, the more likely the cluster resulting in the
 * largest increase in the quality function is to be chosen. A node is
 * merged with a cluster only if both are sufficiently well connected to
 * the rest of the network.
 * </p>
 *
 * @param
 *
 * @return : Number of unique subcommunities formed
 * DO NOT parallelize as it is called within Galois parallel loops
 *
 */
template <typename GraphTy, typename CommTy>
void mergeNodesSubset(GraphTy& graph,
                      std::vector<typename GraphTy::GraphNode>& cluster_nodes,
                      uint64_t comm_id, uint64_t total_degree_wt,
                      CommTy& subcomm_info, double constant_for_second_term) {

  using GNode = typename GraphTy::GraphNode;

  // select set R
  std::vector<GNode> cluster_nodes_to_move;
  for (uint64_t i = 0; i < cluster_nodes.size(); ++i) {
    GNode n      = cluster_nodes[i];
    auto& n_data = graph.getData(n);
    /*
     * Initialize with singleton sub-communities
     */
    EdgeTy nodeEdgeWeightWithinCluster = 0;
    for (auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
      GNode dst      = graph.getEdgeDst(ii);
      EdgeTy edge_wt = graph.getEdgeData(ii, flag_no_lock);
      /*
       * Must include the edge weight of all neighbors excluding self loops
       * belonging to the community comm_id
       */
      if (dst != n && graph.getData(dst).curr_comm_ass == comm_id) {
        nodeEdgeWeightWithinCluster += edge_wt;
      }
    }

    uint64_t node_wt   = n_data.node_wt;
    uint64_t degree_wt = n_data.degree_wt;
    /*
     * Additionally, only nodes that are well connected with
     * the rest of the network are considered for moving.
     * (externalEdgeWeightPerCluster[j] >= clusterWeights[j] * (totalNodeWeight
     * - clusterWeights[j]) * resolution
     */
    if (nodeEdgeWeightWithinCluster >=
        constant_for_second_term * (double)degree_wt *
            ((double)total_degree_wt - (double)degree_wt))
      cluster_nodes_to_move.push_back(n);

    subcomm_info[n].node_wt          = node_wt;
    subcomm_info[n].internal_edge_wt = nodeEdgeWeightWithinCluster;
    subcomm_info[n].size             = 1;
    subcomm_info[n].degree_wt        = degree_wt;
  }

  for (GNode n : cluster_nodes_to_move) {
    auto& n_data = graph.getData(n);
    /*
     * Only consider singleton communities
     */
    if (subcomm_info[n_data.curr_subcomm_ass].size == 1) {
      uint64_t new_subcomm_ass =
          getRandomSubcommunity2(graph, n, subcomm_info, total_degree_wt,
                                 comm_id, constant_for_second_term);

      if ((int64_t)new_subcomm_ass != -1 &&
          new_subcomm_ass != graph.getData(n).curr_subcomm_ass) {
        n_data.curr_subcomm_ass = new_subcomm_ass;

        /*
         * Move the currently selected node to its new cluster and
         * update the clustering statistics.
         */
        galois::atomicAdd(subcomm_info[new_subcomm_ass].node_wt,
                          n_data.node_wt);
        galois::atomicAdd(subcomm_info[new_subcomm_ass].size, (uint64_t)1);
        galois::atomicAdd(subcomm_info[new_subcomm_ass].degree_wt,
                          n_data.degree_wt);

        for (auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
          GNode dst    = graph.getEdgeDst(ii);
          auto edge_wt = graph.getEdgeData(ii, flag_no_lock);
          if (dst != n && graph.getData(dst).curr_comm_ass == comm_id) {
            if (graph.getData(dst).curr_subcomm_ass == new_subcomm_ass) {
              subcomm_info[new_subcomm_ass].internal_edge_wt -= edge_wt;
            } else {
              subcomm_info[new_subcomm_ass].internal_edge_wt += edge_wt;
            }
          }
        }
      }
    }
  }
}

/*
 * Refine the clustering by iterating over the clusters and by
 * trying to split up each cluster into multiple clusters.
 */
template <typename GraphTy, typename CommArrayTy>
void refinePartition(GraphTy& graph, double constant_for_second_term) {

  using GNode     = typename GraphTy::GraphNode;
  using CommArray = CommArrayTy;

  galois::gPrint("Refining\n");

  // set singleton subcommunities
  galois::do_all(
      galois::iterate(graph),
      [&](GNode n) { graph.getData(n).curr_subcomm_ass = n; }, galois::steal());

  // populate nodes into communities
  std::vector<std::vector<GNode>> cluster_bags(2 * graph.size() + 1);
  CommArray comm_info;

  comm_info.allocateBlocked(2 * graph.size() + 1);

  galois::do_all(
      galois::iterate((uint32_t)0, (uint32_t)(2 * graph.size() + 1)),
      [&](uint32_t n) {
        comm_info[n].node_wt   = (uint64_t)0;
        comm_info[n].degree_wt = (uint64_t)0;
      },
      galois::steal());

  for (GNode n : graph) {
    auto& n_data = graph.getData(n, flag_no_lock);
    if (n_data.curr_comm_ass != UNASSIGNED)
      cluster_bags[n_data.curr_comm_ass].push_back(n);

    galois::atomicAdd(comm_info[n_data.curr_comm_ass].node_wt, n_data.node_wt);
    galois::atomicAdd(comm_info[n_data.curr_comm_ass].degree_wt,
                      n_data.degree_wt);
  }

  CommArray subcomm_info;

  subcomm_info.allocateBlocked(graph.size() + 1);

  // call mergeNodesSubset for each community in parallel
  galois::do_all(galois::iterate((uint64_t)0, (uint64_t)graph.size()),
                 [&](uint64_t c) {
                   /*
                    * Only nodes belonging to singleton clusters can be moved to
                    * a different cluster. This guarantees that clusters will
                    * never be split up.
                    */
                   comm_info[c].num_subcomm = 0;
                   if (cluster_bags[c].size() > 1) {
                     // comm_info[c].num_subcomm =
                     mergeNodesSubset<GraphTy, CommArray>(
                         graph, cluster_bags[c], c, comm_info[c].degree_wt,
                         subcomm_info, constant_for_second_term);
                   } else {
                     comm_info[c].num_subcomm = 0;
                   }
                 });
}

/*
 *
 * Graph construction routines to make
 * coarser graphs.
 *
 */
template <typename GraphTy>
void buildNextLevelGraph(GraphTy& graph, GraphTy& graph_next,
                         uint64_t num_unique_clusters) {
  using GNode = typename GraphTy::GraphNode;
  std::cerr << "Inside buildNextLevelGraph\n";

  galois::StatTimer TimerGraphBuild("Timer_Graph_build");
  TimerGraphBuild.start();
  uint32_t num_nodes_next = num_unique_clusters;
  uint64_t num_edges_next = 0; // Unknown right now

  std::vector<std::vector<GNode>> cluster_bags(num_unique_clusters);
  // Comment: Serial separation is better than do_all due to contention
  for (GNode n = 0; n < graph.size(); ++n) {
    auto n_data = graph.getData(n, flag_no_lock);
    if (n_data.curr_comm_ass != UNASSIGNED)
      cluster_bags[n_data.curr_comm_ass].push_back(n);
  }

  std::vector<std::vector<uint32_t>> edges_id(num_unique_clusters);
  std::vector<std::vector<EdgeTy>> edges_data(num_unique_clusters);

  /* First pass to find the number of edges */
  galois::do_all(
      galois::iterate((uint64_t)0, num_unique_clusters),
      [&](uint64_t c) {
        std::map<uint64_t, uint64_t> cluster_local_map;
        uint64_t num_unique_clusters = 0;
        for (auto cb_ii = cluster_bags[c].begin();
             cb_ii != cluster_bags[c].end(); ++cb_ii) {

          assert(graph.getData(*cb_ii, flag_no_lock).curr_comm_ass ==
                 c); // All nodes in this bag must have same cluster id

          for (auto ii = graph.edge_begin(*cb_ii); ii != graph.edge_end(*cb_ii);
               ++ii) {
            GNode dst     = graph.getEdgeDst(ii);
            auto dst_data = graph.getData(dst, flag_no_lock);
            assert(dst_data.curr_comm_ass != UNASSIGNED);
            auto stored_already = cluster_local_map.find(
                dst_data.curr_comm_ass); // Check if it already exists
            if (stored_already != cluster_local_map.end()) {
              edges_data[c][stored_already->second] += graph.getEdgeData(ii);
            } else {
              cluster_local_map[dst_data.curr_comm_ass] = num_unique_clusters;
              edges_id[c].push_back(dst_data.curr_comm_ass);
              edges_data[c].push_back(graph.getEdgeData(ii));
              num_unique_clusters++;
            }
          } // End edge loop
        }
      },
      galois::steal(), galois::loopname("BuildGrah: Find edges"));

  /* Serial loop to reduce all the edge counts */
  std::vector<uint64_t> prefix_edges_count(num_unique_clusters);
  galois::GAccumulator<uint64_t> num_edges_acc;
  galois::do_all(galois::iterate((uint32_t)0, num_nodes_next), [&](uint32_t c) {
    prefix_edges_count[c] = edges_id[c].size();
    num_edges_acc += prefix_edges_count[c];
  });

  num_edges_next = num_edges_acc.reduce();
  for (uint32_t c = 1; c < num_nodes_next; ++c) {
    prefix_edges_count[c] += prefix_edges_count[c - 1];
  }

  assert(prefix_edges_count[num_unique_clusters - 1] == num_edges_next);
  galois::gPrint("#nodes : ", num_nodes_next, ", #edges : ", num_edges_next,
                 "\n");
  std::cerr << "Graph construction started"
            << "\n";
  galois::StatTimer TimerConstructFrom("Timer_Construct_From");
  TimerConstructFrom.start();
  graph_next.constructFrom(num_nodes_next, num_edges_next, prefix_edges_count,
                           edges_id, edges_data);
  TimerConstructFrom.stop();

  TimerGraphBuild.stop();
  galois::gPrint("Graph construction done\n");
}

template <typename GraphTy>
void buildNextLevelGraphSubComm(GraphTy& graph, GraphTy& graph_next,
                                uint64_t num_unique_clusters,
                                std::vector<uint64_t>& original_comm_ass,
                                std::vector<uint64_t>& cluster_node_wt) {
  using GNode = typename GraphTy::GraphNode;

  galois::StatTimer TimerGraphBuild("Timer_Graph_build");
  TimerGraphBuild.start();
  uint32_t num_nodes_next = num_unique_clusters;
  uint64_t num_edges_next = 0; // Unknown right now

  std::vector<std::vector<GNode>> cluster_bags(num_unique_clusters);
  // Comment: Serial separation is better than do_all due to contention
  for (GNode n = 0; n < graph.size(); ++n) {
    auto n_data = graph.getData(n, flag_no_lock);
    original_comm_ass[n_data.curr_subcomm_ass] =
        graph.getData(n_data.curr_comm_ass).curr_subcomm_ass;
    assert(n_data.curr_comm_ass != UNASSIGNED);
    cluster_bags[n_data.curr_subcomm_ass].push_back(n);
    cluster_node_wt[n_data.curr_subcomm_ass] += n_data.node_wt;
  }

  std::vector<std::vector<uint32_t>> edges_id(num_unique_clusters);
  std::vector<std::vector<EdgeTy>> edges_data(num_unique_clusters);

  /* First pass to find the number of edges */
  galois::do_all(
      galois::iterate((uint64_t)0, num_unique_clusters),
      [&](uint64_t c) {
        std::map<uint64_t, uint64_t> cluster_local_map;
        uint64_t num_unique_clusters = 0;
        for (auto cb_ii = cluster_bags[c].begin();
             cb_ii != cluster_bags[c].end(); ++cb_ii) {

          assert(graph.getData(*cb_ii, flag_no_lock).curr_subcomm_ass ==
                 c); // All nodes in this bag must have same cluster id

          for (auto ii = graph.edge_begin(*cb_ii); ii != graph.edge_end(*cb_ii);
               ++ii) {
            GNode dst     = graph.getEdgeDst(ii);
            auto dst_data = graph.getData(dst, flag_no_lock);
            assert(dst_data.curr_subcomm_ass != UNASSIGNED);
            auto stored_already = cluster_local_map.find(
                dst_data.curr_subcomm_ass); // Check if it already exists
            if (stored_already != cluster_local_map.end()) {
              edges_data[c][stored_already->second] += graph.getEdgeData(ii);
            } else {
              cluster_local_map[dst_data.curr_subcomm_ass] =
                  num_unique_clusters;
              edges_id[c].push_back(dst_data.curr_subcomm_ass);
              edges_data[c].push_back(graph.getEdgeData(ii));
              num_unique_clusters++;
            }
          } // End edge loop
        }
      },
      galois::steal(), galois::loopname("BuildGrah: Find edges"));

  /* Serial loop to reduce all the edge counts */
  std::vector<uint64_t> prefix_edges_count(num_unique_clusters);
  galois::GAccumulator<uint64_t> num_edges_acc;
  galois::do_all(galois::iterate((uint32_t)0, num_nodes_next), [&](uint32_t c) {
    prefix_edges_count[c] = edges_id[c].size();
    num_edges_acc += prefix_edges_count[c];
  });

  num_edges_next = num_edges_acc.reduce();
  for (uint32_t c = 1; c < num_nodes_next; ++c) {
    prefix_edges_count[c] += prefix_edges_count[c - 1];
  }

  assert(prefix_edges_count[num_unique_clusters - 1] == num_edges_next);
  galois::gPrint("#nodes : ", num_nodes_next, ", #edges : ", num_edges_next,
                 "\n");
  galois::gPrint("#prefix last : ", prefix_edges_count[num_unique_clusters - 1],
                 "\n");

  std::cerr << "Graph construction started"
            << "\n";

  galois::StatTimer TimerConstructFrom("Timer_Construct_From");
  TimerConstructFrom.start();
  graph_next.constructFrom(num_nodes_next, num_edges_next, prefix_edges_count,
                           edges_id, edges_data);
  TimerConstructFrom.stop();

  std::cout << " c1:" << calConstantForSecondTerm(graph) << "\n";
  TimerGraphBuild.stop();
  galois::gPrint("Graph construction done\n");
}

#endif // CLUSTERING_H
