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

// Graph header, created ::Tue Jun  2 16:02:26 2015

typedef struct _GraphType {
  uint _num_nodes;
  uint _num_edges;
  uint _node_data_size;
  uint _edge_data_size;
  uint _num_owned;
  ulong _global_offset;
  __global NodeData* _node_data;
  __global uint* _out_index;
  __global uint* _out_neighbors;
  __global EdgeData* _out_edge_data;
} GraphType;

uint edge_begin(__global GraphType* graph, uint node) {
  return graph->_out_index[node];
}
uint edge_end(__global GraphType* graph, uint node) {
  return graph->_out_index[node + 1];
}
uint getEdgeDst(__global GraphType* graph, uint nbr) {
  return graph->_out_neighbors[nbr];
}
__global EdgeData* getEdgeData(__global GraphType* graph, uint nbr) {
  return &graph->_out_edge_data[nbr];
}
__global NodeData* getData(__global GraphType* graph, uint node) {
  return &graph->_node_data[node];
}
uint getGID(__global GraphType* graph, uint lid) {
  if (lid < graph->_num_owned) {
    return lid + graph->_global_offset;
  }
  // TODO Finish implementation by adding ghost_map impl.
  return -1;
}
void initialize(__global GraphType* graph, __global uint* mem_pool) {
  uint offset            = 4;
  graph->_num_nodes      = mem_pool[0];
  graph->_num_edges      = mem_pool[1];
  graph->_node_data_size = mem_pool[2];
  graph->_edge_data_size = mem_pool[3];
  graph->_num_owned      = mem_pool[4];
  graph->_global_offset  = mem_pool[6];
  graph->_node_data      = (__global NodeData*)&mem_pool[offset];
  offset += graph->_num_nodes * graph->_node_data_size;
  graph->_out_index = &mem_pool[offset];
  offset += graph->_num_nodes + 1;
  graph->_out_neighbors = &mem_pool[offset];
  offset += graph->_num_edges;
  graph->_out_edge_data = (__global EdgeData*)&mem_pool[offset];
  offset += graph->_num_edges * graph->_edge_data_size;
}

__kernel void initialize_graph_struct(__global uint* res, __global uint* g_meta,
                                      __global NodeData* g_node_data,
                                      __global uint* g_out_index,
                                      __global uint* g_nbr,
                                      __global EdgeData* edge_data) {
  __global GraphType* g = (__global GraphType*)res;
  g->_num_nodes         = g_meta[0];
  g->_num_edges         = g_meta[1];
  g->_node_data_size    = g_meta[2];
  g->_edge_data_size    = g_meta[3];
  g->_num_owned         = g_meta[4];
  g->_global_offset     = g_meta[6];
  g->_node_data         = g_node_data;
  g->_out_index         = g_out_index;
  g->_out_neighbors     = g_nbr;
  g->_out_edge_data     = edge_data;
}

__kernel void initialize_void_graph_struct(__global uint* res,
                                           __global uint* g_meta,
                                           __global NodeData* g_node_data,
                                           __global uint* g_out_index,
                                           __global uint* g_nbr) {
  __global GraphType* g = (__global GraphType*)res;
  g->_num_nodes         = g_meta[0];
  g->_num_edges         = g_meta[1];
  g->_node_data_size    = g_meta[2];
  g->_edge_data_size    = g_meta[3];
  g->_num_owned         = g_meta[4];
  g->_global_offset     = g_meta[6];
  g->_node_data         = g_node_data;
  g->_out_index         = g_out_index;
  g->_out_neighbors     = g_nbr;
}
