/** Forward declarations for cuda host -*- C++ -*-
 * @file HostDecls.h
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Forward declarations for CUDA host
 *
 * @author Roshan Dathathri <roshan@cs.utexas.edu>
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#ifndef __HOST_FORWARD_DECL__
#define __HOST_FORWARD_DECL__
#include <string>

#ifndef LSG_CSR_GRAPH
typedef unsigned int index_type; // GPU kernels choke on size_t 
typedef unsigned int node_data_type;
typedef unsigned int edge_data_type;
#endif

struct MarshalGraph {
  size_t nnodes;  
  size_t nedges;
  unsigned int numOwned; // Number of nodes owned (masters) by this host
  unsigned int beginMaster; // local id of the beginning of master nodes
  unsigned int numNodesWithEdges; // Number of nodes (masters + mirrors) that have outgoing edges 
  int id;
  unsigned numHosts;
  index_type *row_start;
  index_type *edge_dst;
  node_data_type *node_data;
  edge_data_type *edge_data;
  unsigned int *num_master_nodes;
  unsigned int **master_nodes;
  unsigned int *num_mirror_nodes;
  unsigned int **mirror_nodes;

  MarshalGraph() :
    nnodes(0), nedges(0), numOwned(0), beginMaster(0), numNodesWithEdges(0),
    id(-1), numHosts(0),
    row_start(NULL), edge_dst(NULL),
    node_data(NULL), edge_data(NULL),
    num_master_nodes(NULL), master_nodes(NULL),
    num_mirror_nodes(NULL), mirror_nodes(NULL) {}

  ~MarshalGraph() {
    if (!row_start) free(row_start);
    if (!edge_dst) free(edge_dst);
    if (!node_data) free(node_data);
    if (!edge_data) free(edge_data);
    if (!num_master_nodes) free(num_master_nodes);
    if (!master_nodes) {
      for (unsigned i = 0; i < numHosts; ++i) {
        free(master_nodes[i]);
      }
      free(master_nodes);
    }
    if (!num_mirror_nodes) free(num_mirror_nodes);
    if (!mirror_nodes) {
      for (unsigned i = 0; i < numHosts; ++i) {
        free(mirror_nodes[i]);
      }
      free(mirror_nodes);
    }
  }
};

// to determine the GPU device id
int get_gpu_device_id(std::string personality_set, int num_nodes); // defined on the host

struct CUDA_Context; // forward declaration only because rest is dependent on the dist_app

// defined on the device
struct CUDA_Context *get_CUDA_context(int id);
bool init_CUDA_context(struct CUDA_Context *ctx, int device);
void load_graph_CUDA(struct CUDA_Context *ctx, MarshalGraph &g, unsigned num_hosts);
void reset_CUDA_context(struct CUDA_Context *ctx);
#endif
