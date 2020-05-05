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

// Graph header, created ::Tue Jun  2 16:02:26 2015
typedef struct _GraphType {
  uint _num_nodes;
  uint _num_edges;
  uint _node_data_size;
  uint _edge_data_size;
  __global NodeData* _node_data;
  __global uint* _out_index;
  __global uint* _out_neighbors;
  __global EdgeData* _out_edge_data;
} GraphType;
uint in_neighbors_begin(__local GraphType* graph, uint node) { return 0; }
uint in_neighbors_end(__local GraphType* graph, uint node) {
  return graph->_out_index[node + 1] - graph->_out_index[node];
}
uint in_neighbors_next(__local GraphType* graph, uint node) { return 1; }
uint in_neighbors(__local GraphType* graph, uint node, uint nbr) {
  return graph->_out_neighbors[graph->_out_index[node] + nbr];
}
__global EdgeData* in_edge_data(__local GraphType* graph, uint node, uint nbr) {
  return &graph->_out_edge_data[graph->_out_index[node] + nbr];
}
uint out_neighbors_begin(__local GraphType* graph, uint node) { return 0; }
uint out_neighbors_end(__local GraphType* graph, uint node) {
  return graph->_out_index[node + 1] - graph->_out_index[node];
}
uint out_neighbors_next(__local GraphType* graph, uint node) { return 1; }
uint out_neighbors(__local GraphType* graph, uint node, uint nbr) {
  return graph->_out_neighbors[graph->_out_index[node] + nbr];
}
__global EdgeData* out_edge_data(__local GraphType* graph, uint node,
                                 uint nbr) {
  return &graph->_out_edge_data[graph->_out_index[node] + nbr];
}
__global NodeData* node_data(__local GraphType* graph, uint node) {
  return &graph->_node_data[node];
}
void initialize(__local GraphType* graph, __global uint* mem_pool) {
  uint offset            = 4;
  graph->_num_nodes      = mem_pool[0];
  graph->_num_edges      = mem_pool[1];
  graph->_node_data_size = mem_pool[2];
  graph->_edge_data_size = mem_pool[3];
  graph->_node_data      = (__global NodeData*)&mem_pool[offset];
  offset += graph->_num_nodes * graph->_node_data_size;
  graph->_out_index = &mem_pool[offset];
  offset += graph->_num_nodes + 1;
  graph->_out_neighbors = &mem_pool[offset];
  offset += graph->_num_edges;
  graph->_out_edge_data = (__global EdgeData*)&mem_pool[offset];
  offset += graph->_num_edges * graph->_edge_data_size;
}
