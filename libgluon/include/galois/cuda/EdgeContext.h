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
 * @file cuda/EdgeContext.h
 *
 * Contains definition of CUDA context structures.
 *
 * @todo document this file
 */

#ifndef __EDGE_CONTEXT__
#define __EDGE_CONTEXT__

#pragma once
#include <cuda.h>
#include "gg.h"
#pragma once
#include "galois/cuda/EdgeHostDecls.h"

struct CUDA_Context_Shared_Edges {
  unsigned int* num_edges;         // per host
  DeviceOnly<unsigned int>* edges; // per host
};

struct CUDA_Context_Common_Edges {
  int device;
  int id;
  unsigned int numOwned;    // Number of nodes owned (masters) by this host
  unsigned int beginMaster; // local id of the beginning of master nodes
  unsigned int numNodesWithEdges; // Number of nodes (masters + mirrors) that
                                  // have outgoing edges
  CSRGraphTy gg;
  struct CUDA_Context_Shared_Edges master;
  struct CUDA_Context_Shared_Edges mirror;
  DeviceOnly<unsigned int> offsets; // union across master/mirror of all hosts
  Shared<DynamicBitset> is_updated; // union across master/mirror of all hosts
};

template <typename Type>
struct CUDA_Context_Field_Edges {
  Shared<Type> data;
  Shared<DynamicBitset> is_updated; // size of edges
  DeviceOnly<Type> shared_data;     // union across master/mirror of all hosts
};

bool init_CUDA_context_common_edges(struct CUDA_Context_Common_Edges* ctx,
                                    int device) {
  struct cudaDeviceProp dev;
  if (device == -1) {
    check_cuda(cudaGetDevice(&device));
  } else {
    int count;
    check_cuda(cudaGetDeviceCount(&count));
    if (device > count) {
      fprintf(stderr, "Error: Out-of-range GPU %d specified (%d total GPUs)",
              device, count);
      return false;
    }
    check_cuda(cudaSetDevice(device));
  }
  ctx->device = device;
  check_cuda(cudaGetDeviceProperties(&dev, device));
  printf("[%d] Using GPU %d: %s\n", ctx->id, device, dev.name);
  return true;
}

void load_graph_CUDA_common_edges(struct CUDA_Context_Common_Edges* ctx,
                                  EdgeMarshalGraph& g, unsigned num_hosts,
                                  bool LoadProxyEdges = true) {
  CSRGraphTy graph;
  ctx->numOwned          = g.numOwned;
  ctx->beginMaster       = g.beginMaster;
  ctx->numNodesWithEdges = g.numNodesWithEdges;
  assert(ctx->id == g.id);

  size_t mem_usage = ((g.nnodes + 1) + g.nedges) * sizeof(index_type) +
                     (g.nnodes) * sizeof(node_data_type);
  if (!g.edge_data)
    mem_usage += (g.nedges) * sizeof(edge_data_type);
  printf("[%d] Host memory for graph: %3u MB\n", ctx->id, mem_usage / 1048756);

  // copy the graph to the GPU
  graph.nnodes    = g.nnodes;
  graph.nedges    = g.nedges;
  graph.row_start = g.row_start;
  graph.edge_dst  = g.edge_dst;
  graph.node_data = g.node_data;
  graph.edge_data = g.edge_data;
  graph.copy_to_gpu(ctx->gg);

  if (LoadProxyEdges) {
    size_t max_shared_size = 0; // for union across master/mirror of all hosts
    ctx->master.num_edges =
        (unsigned int*)calloc(num_hosts, sizeof(unsigned int));
    memcpy(ctx->master.num_edges, g.num_master_edges,
           sizeof(unsigned int) * num_hosts);
    ctx->master.edges = (DeviceOnly<unsigned int>*)calloc(
        num_hosts, sizeof(Shared<unsigned int>));
    for (uint32_t h = 0; h < num_hosts; ++h) {
      if (ctx->master.num_edges[h] > 0) {
        ctx->master.edges[h].alloc(ctx->master.num_edges[h]);
        ctx->master.edges[h].copy_to_gpu(g.master_edges[h],
                                         ctx->master.num_edges[h]);
      }
      if (ctx->master.num_edges[h] > max_shared_size) {
        max_shared_size = ctx->master.num_edges[h];
      }
    }
    ctx->mirror.num_edges =
        (unsigned int*)calloc(num_hosts, sizeof(unsigned int));
    memcpy(ctx->mirror.num_edges, g.num_mirror_edges,
           sizeof(unsigned int) * num_hosts);
    ctx->mirror.edges = (DeviceOnly<unsigned int>*)calloc(
        num_hosts, sizeof(Shared<unsigned int>));
    for (uint32_t h = 0; h < num_hosts; ++h) {
      if (ctx->mirror.num_edges[h] > 0) {
        ctx->mirror.edges[h].alloc(ctx->mirror.num_edges[h]);
        ctx->mirror.edges[h].copy_to_gpu(g.mirror_edges[h],
                                         ctx->mirror.num_edges[h]);
      }
      if (ctx->mirror.num_edges[h] > max_shared_size) {
        max_shared_size = ctx->mirror.num_edges[h];
      }
    }
    ctx->offsets.alloc(max_shared_size);
    ctx->is_updated.alloc(1);
    ctx->is_updated.cpu_wr_ptr()->alloc(max_shared_size);
  }
  // printf("[%u] load_graph_GPU: %u owned nodes of total %u resident, %lu
  // edges\n", ctx->id, ctx->nowned, graph.nnodes, graph.nedges);
}

size_t mem_usage_CUDA_common_edges(EdgeMarshalGraph& g, unsigned num_hosts) {
  size_t mem_usage       = 0;
  size_t max_shared_size = 0; // for union across master/mirror of all hosts
  mem_usage += num_hosts * sizeof(unsigned int);
  mem_usage += num_hosts * sizeof(Shared<unsigned int>);
  for (uint32_t h = 0; h < num_hosts; ++h) {
    if (g.num_master_edges[h] > 0) {
      mem_usage += g.num_master_edges[h] * sizeof(unsigned int);
    }
    if (g.num_master_edges[h] > max_shared_size) {
      max_shared_size = g.num_master_edges[h];
    }
  }
  mem_usage += num_hosts * sizeof(unsigned int);
  mem_usage += num_hosts * sizeof(Shared<unsigned int>);
  for (uint32_t h = 0; h < num_hosts; ++h) {
    if (g.num_mirror_edges[h] > 0) {
      mem_usage += g.num_mirror_edges[h] * sizeof(unsigned int);
    }
    if (g.num_mirror_edges[h] > max_shared_size) {
      max_shared_size = g.num_mirror_edges[h];
    }
  }
  mem_usage += max_shared_size * sizeof(unsigned int);
  mem_usage += ((max_shared_size + 63) / 64) * sizeof(unsigned long long int);
  return mem_usage;
}

template <typename Type>
void load_graph_CUDA_field_edges(struct CUDA_Context_Common_Edges* ctx,
                                 struct CUDA_Context_Field_Edges<Type>* field,
                                 unsigned num_hosts) {
  field->data.alloc(ctx->gg.nedges);
  size_t max_shared_size = 0; // for union across master/mirror of all hosts
  for (uint32_t h = 0; h < num_hosts; ++h) {
    if (ctx->master.num_edges[h] > max_shared_size) {
      max_shared_size = ctx->master.num_edges[h];
    }
  }
  for (uint32_t h = 0; h < num_hosts; ++h) {
    if (ctx->mirror.num_edges[h] > max_shared_size) {
      max_shared_size = ctx->mirror.num_edges[h];
    }
  }
  field->shared_data.alloc(max_shared_size);
  field->is_updated.alloc(1);
  field->is_updated.cpu_wr_ptr()->alloc(ctx->gg.nedges);
}

template <typename Type>
size_t mem_usage_CUDA_field_edges(struct CUDA_Context_Field_Edges<Type>* field,
                                  EdgeMarshalGraph& g, unsigned num_hosts) {
  size_t mem_usage = 0;
  mem_usage += g.nedges * sizeof(Type);
  size_t max_shared_size = 0; // for union across master/mirror of all hosts
  for (uint32_t h = 0; h < num_hosts; ++h) {
    if (g.num_master_edges[h] > max_shared_size) {
      max_shared_size = g.num_master_edges[h];
    }
  }
  for (uint32_t h = 0; h < num_hosts; ++h) {
    if (g.num_mirror_edges[h] > max_shared_size) {
      max_shared_size = g.num_mirror_edges[h];
    }
  }
  mem_usage += max_shared_size * sizeof(Type);
  mem_usage += ((g.nedges + 63) / 64) * sizeof(unsigned long long int);
  return mem_usage;
}
#endif
