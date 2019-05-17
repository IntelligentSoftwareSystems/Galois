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


#include <boost/dynamic_bitset.hpp>
#include <stdlib.h>
// returns the host
#if 0
      bool is_assigned(uint64_t GID){
        if(node_mapping[GID] != ~0)
          return  true;
        else
          return false;
      }

      void assign_node(uint64_t GID, unsigned host){
        node_mapping[GID].push_back(host);
      }
#endif

uint32_t random_edge_assignment(uint64_t src, uint64_t dst,
                                galois::VecBool& bitset, uint32_t numHosts) {

  uint32_t rand_number = rand() % numHosts;
  return rand_number;
}

uint32_t balanced_edge_assignment(uint64_t src, uint64_t dst,
                                  galois::VecBool& bitset, uint32_t numHosts,
                                  std::vector<uint64_t>& numEdges_per_host) {
  uint32_t min_load_host = ~0;
  uint64_t load          = 0;
  uint32_t minEdge       = ~0;
  uint64_t minEdge_load  = ~0;

  for (uint32_t k = 0; k < numHosts; ++k) {
    if (numEdges_per_host[k] < minEdge_load) {
      minEdge      = k;
      minEdge_load = numEdges_per_host[k];
    }
    if (bitset.is_set(src, k) && bitset.is_set(dst, k)) {
      // both have common host.
      if (min_load_host == (uint32_t)(~0)) {
        min_load_host = k;
        load          = numEdges_per_host[k];
      } else if (numEdges_per_host[k] < load) {
        min_load_host = k;
        load          = numEdges_per_host[k];
      }
    }
  }
  if (min_load_host != (uint32_t)(~0))
    return min_load_host;
  else
    return minEdge;
}

uint32_t balanced_edge_assignment2(uint64_t src, uint64_t dst,
                                   galois::VecBool& bitset, uint32_t numHosts,
                                   std::vector<uint64_t>& numEdges_per_host) {

  std::vector<uint32_t> intersection_vec, union_vec;
  for (uint32_t k = 0; k < numHosts; ++k) {
    if (bitset.is_set(src, k) && bitset.is_set(dst, k)) {
      intersection_vec.push_back(k);
    }
    if (bitset.is_set(src, k) || bitset.is_set(dst, k)) {
      union_vec.push_back(k);
    }
  }

  // both have common host.
  if (intersection_vec.size() > 0) {
    auto min_load_host = intersection_vec[0];
    for (auto h : intersection_vec) {
      if (numEdges_per_host[h] < numEdges_per_host[min_load_host]) {
        min_load_host = h;
      }
    }
    return min_load_host;
  }

  // must be assigned to a host where src or dst is.
  if (union_vec.size() > 0) {
    auto min_load_host = union_vec[0];
    for (auto h : union_vec) {
      if (numEdges_per_host[h] < numEdges_per_host[min_load_host]) {
        min_load_host = h;
      }
    }
    return min_load_host;
  }

  // not assigned to any host.
  auto src_host = src % numHosts;
  auto dst_host = dst % numHosts;

  if (numEdges_per_host[src_host] < numEdges_per_host[dst_host]) {
    return src_host;
  } else {
    return dst_host;
  }
  return 0;
}

#if 0
/* POLICY */
if(!is_assigned(src) && !is_assigned(dst)){
  auto src_host = src%base_DistGraph::numHosts;
  auto dst_host = dst%base_DistGraph::numHosts;

            if(num_edges_assigned[src_host].size() < num_edges_assigned[dst_host].size()){
              assign_node(src,src_host);
              assign_node(dst,src_host);
              num_edges_assigned[src_host] += 1;
            }
            else {
              assign_node(src,dst_host);
              assign_node(dst,dst_host);
              num_edges_assigned[dst_host] += 1;
            }
          }

          /***** atleast one of src or dst belongs to a host *****/
          else {
            if(is_assigned(src) && !is_assigned(dst)){
              auto src_host = get_assigned_host(src);
              auto dst_host = dst%base_DistGraph::numHosts;
              if(num_edges_assigned[src_host].size() < num_edges_assigned[dst_host].size()){
                assign_node(dst, src_host);
              }
              else {
                assign_node(dst, dst_host);
                master_mapping[src_host] = src;
                mirror_mapping[dst_host] = src;
              }
          
            if(!is_assigned(src) && is_assigned(dst)){
              auto src_host = src%base_DistGraph::numHosts;
              auto dst_host = get_assigned_host(dst);
          
          
          }

        }
      }

#endif
