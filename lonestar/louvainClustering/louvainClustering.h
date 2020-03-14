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


#ifndef LOUVAIN_CLUSTERING_H
#define LOUVAIN_CLUSTERING_H






/*
 * Typedefs
 */

constexpr static const uint64_t INF_VAL = std::numeric_limits<uint64_t>::max() / 2 - 1;

constexpr galois::MethodFlag flag_no_lock = galois::MethodFlag::UNPROTECTED;
constexpr galois::MethodFlag flag_read_lock = galois::MethodFlag::READ;
constexpr galois::MethodFlag flag_write_lock = galois::MethodFlag::WRITE;

typedef galois::LargeArray<uint64_t> largeArray;

//Maintain community information
struct Comm {
  std::atomic<uint64_t> size;
  std::atomic<uint64_t> degree_wt;
};

typedef galois::LargeArray<Comm> CommArray;

//Graph Node information
struct Node{
  uint64_t prev_comm_ass;
  uint64_t curr_comm_ass;
  uint64_t degree_wt;
  uint64_t cluster_wt_internal;
  int64_t colorId;
};

typedef uint32_t EdgeTy;
using Graph =
    galois::graphs::LC_CSR_Graph<Node, EdgeTy>::with_no_lockable<false>::type::with_numa_alloc<true>::type;

using GNode = Graph::GraphNode;




uint64_t vertexFollowing(Graph& graph, largeArray& clusters){

  //Initialize each node to its own cluster
  galois::do_all(galois::iterate(graph),
                 [&graph](GNode n) { graph.getData(n).curr_comm_ass = n; });

  //Remove isolated and degree-one nodes
  galois::GAccumulator<uint64_t> isolatedNodes;
  galois::do_all(galois::iterate(graph),
                  [&](GNode n) {
                    uint64_t degree = std::distance(graph.edge_begin(n, galois::MethodFlag::UNPROTECTED),
                                                     graph.edge_end(n, galois::MethodFlag::UNPROTECTED));
                    if(degree == 0) {
                      isolatedNodes += 1;
                      clusters[n] = -1;
                    } else {
                      if(degree == 1) {
                        //Check if the destination has degree greater than one
                        auto dst = graph.getEdgeDst(graph.edge_end(n, galois::MethodFlag::UNPROTECTED));
                        uint64_t dst_degree = std::distance(graph.edge_begin(dst, galois::MethodFlag::UNPROTECTED),
                                                         graph.edge_end(dst, galois::MethodFlag::UNPROTECTED));
                        if((dst_degree > 1 || (n > dst))){
                          isolatedNodes += 1;
                          clusters[n] = graph.getData(dst).curr_comm_ass;
                        }
                      }
                    }
                  });
  //The number of isolated nodes that can be removed
  return isolatedNodes.reduce();
}



void sumVertexDegreeWeight(Graph& graph, CommArray& c_info) {
//void sumVertexDegreeWeight(Graph& graph, std::vector<Comm>& c_info) {
  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  uint64_t total_weight = 0;
                  auto &n_data = graph.getData(n);
                  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii){
                     total_weight += graph.getEdgeData(ii, flag_no_lock);
                  }
                  n_data.degree_wt = total_weight;
                  c_info[n].degree_wt = total_weight;
                  //galois::gPrint(n, " : ", c_info[n].degree_wt.load(), "\n");
                  c_info[n].size = 1;
                });
}

double calConstantForSecondTerm(Graph& graph){
  galois::GAccumulator<uint64_t> local_weight;
  galois::do_all(galois::iterate(graph),
                [&graph, &local_weight](GNode n){
                  local_weight += graph.getData(n).degree_wt;
                });
  /* This is twice since graph is symmetric */
  uint64_t total_edge_weight_twice = local_weight.reduce();
  return 1/(double)total_edge_weight_twice;
}


uint64_t maxModularity(std::map<uint64_t, uint64_t> &cluster_local_map, std::vector<uint64_t> &counter, uint64_t self_loop_wt,
                       //std::vector<Comm>&c_info, uint64_t degree_wt, uint64_t sc, double constant) {
                       CommArray &c_info, uint64_t degree_wt, uint64_t sc, double constant) {

  uint64_t max_index = sc; // Assign the intial value as self community
  double cur_gain = 0;
  double max_gain = 0;
  double eix = counter[0] - self_loop_wt;
  double ax = c_info[sc].degree_wt - degree_wt;
  double eiy = 0;
  double ay = 0;

  auto stored_already = cluster_local_map.begin();
  do {
    if(sc != stored_already->first) {
      ay = c_info[stored_already->first].degree_wt; // Degree wt of cluster y
      eiy = counter[stored_already->second]; // Total edges incident on cluster y
      //cur_gain = 2 * (eiy - eix) - 2 * degree_wt * (ay - ax) * constant;
      //From the paper: Verbatim
      cur_gain = 2 * constant * (eiy - eix) + 2 * degree_wt * (ax - ay) * constant * constant;

      if( (cur_gain > max_gain) ||  ((cur_gain == max_gain) && (cur_gain != 0) && (stored_already->first < max_index))) {
        max_gain = cur_gain;
        max_index = stored_already->first;
      }
    }
    stored_already++; // Explore next cluster
  } while (stored_already != cluster_local_map.end());

  //galois::gPrint("Max Gain : ", max_gain, "\n");
  //if(max_gain < 1e-3 || (c_info[max_index].size == 1 && c_info[sc].size == 1 && max_index > sc)) {
  if((c_info[max_index].size == 1 && c_info[sc].size == 1 && max_index > sc)) {
    max_index = sc;
  }

  assert(max_gain >= 0);
  return max_index;
}

double calModularity(Graph& graph) {
  CommArray c_info; // Community info
  CommArray c_update; // Used for updating community

  /* Variables needed for Modularity calculation */
  uint64_t total_edge_wt_twice;
  double constant_for_second_term;
  double mod = -1;

  largeArray cluster_wt_internal;
  //std::vector<uint64_t> cluster_wt_internal(graph.size());


  /*** Initialization ***/
  c_info.allocateBlocked(graph.size());
  c_update.allocateBlocked(graph.size());
  cluster_wt_internal.allocateBlocked(graph.size());

  /* Calculate the weighted degree sum for each vertex */
  sumVertexDegreeWeight(graph, c_info);

  /* Compute the total weight (2m) and 1/2m terms */
  constant_for_second_term = calConstantForSecondTerm(graph);


   /* Calculate the overall modularity */
  double e_xx = 0;
  galois::GAccumulator<double> acc_e_xx;
  double a2_x = 0;
  galois::GAccumulator<double> acc_a2_x;

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  cluster_wt_internal[n] = 0;
                });

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  auto n_data = graph.getData(n);
                  for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
                    if(graph.getData(graph.getEdgeDst(ii)).curr_comm_ass == n_data.curr_comm_ass) {
                      cluster_wt_internal[n] += graph.getEdgeData(ii);
                    }
                  }
                });

  galois::do_all(galois::iterate(graph),
                [&](GNode n) {
                  acc_e_xx += cluster_wt_internal[n];
                  acc_a2_x += (c_info[n].degree_wt) * (c_info[n].degree_wt);
                });


  e_xx = acc_e_xx.reduce();
  a2_x = acc_a2_x.reduce();

  //galois::gPrint("e_xx : ", e_xx, " ,constant_for_second_term : ", constant_for_second_term, " a2_x : ", a2_x, "\n");
  mod = e_xx * (double)constant_for_second_term - a2_x * (double)constant_for_second_term * (double)constant_for_second_term;
  galois::gPrint("Final Stats: ", " Number of clusters:  ", graph.size() , " Modularity: ", mod, "\n");

  return mod;
}

uint64_t renumberClustersContiguously(Graph &graph) {

  std::map<uint64_t, uint64_t> cluster_local_map;
  uint64_t num_unique_clusters = 0;

  for (GNode n = 0; n < graph.size(); ++n){
    auto& n_data = graph.getData(n, flag_no_lock);
    //if(n_data.curr_comm_ass < INF_VAL) {
    if(n_data.curr_comm_ass != -1) {
      assert(n_data.curr_comm_ass < graph.size());
      auto stored_already = cluster_local_map.find(n_data.curr_comm_ass);
     if(stored_already != cluster_local_map.end()){
      n_data.curr_comm_ass = stored_already->second;
     } else {
      cluster_local_map[n_data.curr_comm_ass] = num_unique_clusters;
      n_data.curr_comm_ass = num_unique_clusters;
      num_unique_clusters++;
     }
    }
  }

  return num_unique_clusters;
}

void printGraph(Graph& graph){
  for(GNode n = 0; n < graph.size(); ++n) {
    for(auto ii = graph.edge_begin(n); ii != graph.edge_end(n); ++ii) {
      galois::gPrint(n, " --> ", graph.getEdgeDst(ii), " , ", graph.getEdgeData(ii), "\n");
    }
  }
}

#endif //LOUVAIN_CLUSTERING_H
