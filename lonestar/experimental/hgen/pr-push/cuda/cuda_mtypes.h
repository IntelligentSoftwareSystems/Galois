#pragma once

// types to marshal Galois types out of Galois.

// required because of nvcc does not support clang on Linux.

#ifndef LSG_CSR_GRAPH
typedef unsigned int index_type; // GPU kernels choke on size_t
typedef unsigned int node_data_type;
typedef unsigned int edge_data_type;
#endif

struct MarshalGraph {
  size_t nnodes;
  size_t nedges;
  size_t nowned;
  size_t g_offset;
  int id;
  index_type* row_start;
  index_type* edge_dst;
  node_data_type* node_data;
  edge_data_type* edge_data;
};
