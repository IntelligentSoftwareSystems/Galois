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
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"

#include <fstream>

/*
 * Typedefs
 */
constexpr static const uint64_t UNASSIGNED =
    std::numeric_limits<uint64_t>::max();

constexpr galois::MethodFlag flag_no_lock    = galois::MethodFlag::UNPROTECTED;
constexpr galois::MethodFlag flag_read_lock  = galois::MethodFlag::READ;
constexpr galois::MethodFlag flag_write_lock = galois::MethodFlag::WRITE;

typedef galois::LargeArray<uint64_t> largeArray;

struct EdgeTy {

  float weight;
};

typedef galois::LargeArray<float> largeArrayEdgeTy;

struct Node {
  uint64_t prev_comm_ass;
  uint64_t curr_comm_ass;
  uint64_t degree_wt;
  uint64_t distance;
  uint32_t part;
  uint64_t weight;
  uint32_t degree;
  bool isCut;

  void setPart(uint32_t p) { part = p; }

  uint32_t getPart() { return part; }

  void setWeight(uint64_t w) { weight = w; }

  uint64_t getWeight() { return weight; }

  void setDegree(uint32_t d) { degree = d; }

  uint32_t getDegree() { return degree; }

  void setIsCut(bool b) { isCut = b; }

  bool getIsCut() { return isCut; }
};

using Graph = galois::graphs::LC_CSR_Graph<Node, EdgeTy>::with_no_lockable<
    false>::type::with_numa_alloc<true>::type;
using GNode = Graph::GraphNode;

// prints the graph characteristics
template <typename GraphTy>
void printGraphCharateristics(GraphTy& graph) {

  galois::gPrint("/******************************************/\n");
  galois::gPrint("/************ Graph Properties ************/\n");
  galois::gPrint("/******************************************/\n");
  galois::gPrint("Number of Nodes: ", graph.size(), "\n");
  galois::gPrint("Number of Edges: ", graph.sizeEdges(), "\n");
}

// checks if the non-leaf node is a boundary node
template <typename GraphTy>
bool checkIsCut(GraphTy& graph, GNode n) {

  bool part = graph.getData(n).getPart();

  for (auto edge : graph.edges(n)) {
    auto dst = graph.getEdgeDst(edge);
    if (graph.getData(dst).getDegree() > 1 &&
        graph.getData(dst).getPart() != part)
      return true;
  }

  return false;
}

// computes degree weight
template <typename GraphTy, typename CommArrayTy>
void sumVertexDegreeWeight(GraphTy& graph, CommArrayTy& c_info) {
  using GNode = typename GraphTy::GraphNode;
  galois::do_all(galois::iterate(graph), [&](GNode n) {
    uint64_t total_weight = 0;
    auto& n_data          = graph.getData(n);
    for (auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
      total_weight += graph.getEdgeData(ii, flag_no_lock).weight;
    }
    n_data.degree_wt    = total_weight;
    c_info[n].degree_wt = total_weight;
    c_info[n].size      = 1;
  });
}

// computes 1/2m term
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

// computes the best community to assign to
template <typename CommArrayTy>
uint64_t maxEdgeCut(std::map<uint64_t, uint64_t>& cluster_local_map,
                    std::vector<EdgeTy>& counter, uint64_t self_loop_wt,
                    CommArrayTy& c_info, uint64_t sc, double constant) {

  uint64_t max_index = sc; // Assign the intial value as self community
  double cur_gain    = 0;
  double max_gain    = 0;
  double eix         = counter[0].weight - self_loop_wt;
  double eiy         = 0;

  auto stored_already = cluster_local_map.begin();
  do {
    if (sc != stored_already->first) {
      eiy = counter[stored_already->second]
                .weight; // Total edges incident on cluster y
      cur_gain = 2 * constant * (eiy - eix);

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

// computes the total modularity
template <typename GraphTy>
double calModularityDelay(GraphTy& graph, double& e_xx,
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

  galois::do_all(galois::iterate(graph),
                 [&](GNode n) { cluster_wt_internal[n] = 0; });

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    for (auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
      if (local_target[graph.getEdgeDst(ii)] == local_target[n]) {
        cluster_wt_internal[n] += graph.getEdgeData(ii).weight;
      }
    }
  });

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    acc_e_xx += cluster_wt_internal[n];
  });

  e_xx = acc_e_xx.reduce();

  mod = e_xx * (double)constant_for_second_term;

  return mod;
}

// renumber the clusters/communities
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

// print the partitions
template <typename GraphTy>
void printPartitions(GraphTy& graph, std::string output_CID_filename) {
  using GNode = typename GraphTy::GraphNode;
  std::ofstream outputFile(output_CID_filename, std::ofstream::out);
  for (GNode n = 0; n < graph.size(); ++n) {
    outputFile << n << "  " << graph.getData(n).getPart() << "\n";
    // outputFile << graph.getData(n).curr_comm_ass << "\n";
  }
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

  galois::LargeArray<uint64_t> weight;

  weight.allocateBlocked(num_nodes_next + 1);

  for (GNode n = 0; n < num_nodes_next + 1; ++n)
    weight[n] = 0;

  std::vector<std::vector<GNode>> cluster_bags(num_unique_clusters);
  // Comment: Serial separation is better than do_all due to contention
  for (GNode n = 0; n < graph.size(); ++n) {
    auto n_data = graph.getData(n, flag_no_lock);
    if (n_data.curr_comm_ass != UNASSIGNED) {
      cluster_bags[n_data.curr_comm_ass].push_back(n);
      weight[n_data.curr_comm_ass] += graph.getData(n).getWeight();
    }
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
                                         // std::cout << "data: " <<
                                         // graph.getEdgeData(ii) << std::endl;
            if (stored_already != cluster_local_map.end()) {
              edges_data[c][stored_already->second].weight +=
                  graph.getEdgeData(ii).weight;
            } else {
              cluster_local_map[dst_data.curr_comm_ass] = num_unique_clusters;
              edges_id[c].push_back(dst_data.curr_comm_ass);
              EdgeTy edge;
              edge.weight = graph.getEdgeData(ii).weight;
              edges_data[c].push_back(edge);
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

  galois::do_all(galois::iterate(graph_next),
                 [&](GNode n) { graph_next.getData(n).setWeight(weight[n]); });

  weight.destroy();
  weight.deallocate();
  TimerConstructFrom.stop();

  TimerGraphBuild.stop();
  galois::gPrint("Graph construction done\n");
}

// computes the edge cut
template <typename GraphTy>
uint64_t computingCut(GraphTy& g) {
  using GNode = typename GraphTy::GraphNode;

  galois::InsertBag<GNode> bag;
  galois::GAccumulator<unsigned> edgecut;
  galois::do_all(
      galois::iterate(g),
      [&](GNode n) {
        uint32_t part = g.getData(n).part;
        for (auto edge : g.edges(n)) {
          auto dst          = g.getEdgeDst(edge);
          uint32_t part_dst = g.getData(dst).getPart();
          if (part != part_dst) {

            edgecut += g.getEdgeData(edge).weight;
          }
        }
      },
      galois::loopname("cutsize"));

  return edgecut.reduce();
}

// assign leaf nodes to the same partition as their neighbor
template <typename GraphTy>
void fixLeafNodes(GraphTy& graph) {

  using GNode = typename GraphTy::GraphNode;

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    uint32_t size = 0;
    uint32_t part;

    for (auto edge : graph.edges(n)) {
      auto dst = graph.getEdgeDst(edge);

      if (dst == n)
        continue;
      size++;
      if (size > 1)
        break;
      part = graph.getData(dst).getPart();
    }

    if (size == 1)
      graph.getData(n).setPart(part);
    //}
  });
}

// computes an initial bi-partition
void partition(Graph&, double);

// compute the degrees
template <typename GraphTy>
void computeDegrees(GraphTy& graph) {

  using GNode = typename GraphTy::GraphNode;

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    uint32_t size = 0;

    for (auto e1 : graph.edges(n)) {
      auto dst1 = graph.getEdgeDst(e1);

      if (dst1 == n)
        continue;
      size++;
    }

    graph.getData(n).setDegree(size);
  });
}

// extend the boundary
void extendBoundary(Graph&, uint32_t);

// output partition sizes
template <typename GraphTy>
void outputPartitionStats(GraphTy& graph) {

  galois::GAccumulator<uint64_t> zeros, ones;

  galois::do_all(galois::iterate(graph), [&](GNode n) {
    int part = graph.getData(n).getPart();
    if (part == 0)
      zeros += graph.getData(n).getWeight();
    else
      ones += graph.getData(n).getWeight();
  });

  std::cout << "ones:" << ones.reduce() << " zeros: " << zeros.reduce()
            << std::endl;
}
#endif // CLUSTERING_H
