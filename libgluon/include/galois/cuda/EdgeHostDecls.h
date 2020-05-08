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

/**
 * @file EdgeHostDecls.h
 *
 * Contains forward declarations and the definition of the EdgeMarshalGraph
 * class, which is used to marshal a graph to GPUs.
 *
 * @todo document this file
 */

#ifndef __HOST_EDGEMARSHAL_FORWARD_DECL__
#define __HOST_EDGEMARSHAL_FORWARD_DECL__
#include <string>

#ifndef LSG_CSR_GRAPH
typedef unsigned int index_type; // GPU kernels choke on size_t
typedef unsigned int node_data_type;
typedef unsigned edge_data_type;
#endif

struct EdgeMarshalGraph {
  size_t nnodes;
  size_t nedges;
  unsigned int numOwned;    // Number of nodes owned (masters) by this host
  unsigned int beginMaster; // local id of the beginning of master nodes
  unsigned int numNodesWithEdges; // Number of nodes (masters + mirrors) that
                                  // have outgoing edges
  int id;
  unsigned numHosts;
  index_type* row_start;
  index_type* edge_dst;
  node_data_type* node_data;
  edge_data_type* edge_data;
  unsigned int* num_master_edges;
  unsigned int** master_edges;
  unsigned int* num_mirror_edges;
  unsigned int** mirror_edges;

  EdgeMarshalGraph()
      : nnodes(0), nedges(0), numOwned(0), beginMaster(0), numNodesWithEdges(0),
        id(-1), numHosts(0), row_start(NULL), edge_dst(NULL), node_data(NULL),
        edge_data(NULL), num_master_edges(NULL), master_edges(NULL),
        num_mirror_edges(NULL), mirror_edges(NULL) {}

  ~EdgeMarshalGraph() {
    if (!row_start)
      free(row_start);
    if (!edge_dst)
      free(edge_dst);
    if (!node_data)
      free(node_data);
    if (!edge_data)
      free(edge_data);
    if (!num_master_edges)
      free(num_master_edges);
    if (master_edges != NULL) {
      for (unsigned i = 0; i < numHosts; ++i) {
        free(master_edges[i]);
      }
      free(master_edges);
    }
    if (!num_mirror_edges)
      free(num_mirror_edges);
    if (mirror_edges != NULL) {
      for (unsigned i = 0; i < numHosts; ++i) {
        free(mirror_edges[i]);
      }
      free(mirror_edges);
    }
  }
};

// to determine the GPU device id
int get_gpu_device_id(std::string personality_set,
                      int num_nodes); // defined on the host

struct CUDA_Context; // forward declaration only because rest is dependent on
                     // the dist_app

// defined on the device
struct CUDA_Context* get_CUDA_context(int id);
bool init_CUDA_context(struct CUDA_Context* ctx, int device);
void load_graph_CUDA(struct CUDA_Context* ctx, EdgeMarshalGraph& g,
                     unsigned num_hosts);
void reset_CUDA_context(struct CUDA_Context* ctx);
#endif
