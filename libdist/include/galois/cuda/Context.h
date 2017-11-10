#pragma once
#include <cuda.h>
#include "gg.h"
#include "galois/cuda/HostDecls.h"

struct CUDA_Context_Shared {
  unsigned int *num_nodes; // per host
  DeviceOnly<unsigned int> *nodes; // per host
};

struct CUDA_Context_Common {
  int device;
  int id;
  unsigned int numOwned; // Number of nodes owned (masters) by this host
  unsigned int beginMaster; // local id of the beginning of master nodes
  unsigned int numNodesWithEdges; // Number of nodes (masters + mirrors) that have outgoing edges 
  CSRGraphTy gg;
  struct CUDA_Context_Shared master;
  struct CUDA_Context_Shared mirror;
  DeviceOnly<unsigned int> offsets; // union across master/mirror of all hosts 
  Shared<DynamicBitset> is_updated; // union across master/mirror of all hosts 
};

template<typename Type>
struct CUDA_Context_Field { 
  Shared<Type> data;
  Shared<DynamicBitset> is_updated;
  DeviceOnly<Type> shared_data; // union across master/mirror of all hosts
};

bool init_CUDA_context_common(struct CUDA_Context_Common *ctx, int device) {
  struct cudaDeviceProp dev;
  if(device == -1) {
    check_cuda(cudaGetDevice(&device));
  } else {
    int count;
    check_cuda(cudaGetDeviceCount(&count));
    if(device > count) {
      fprintf(stderr, "Error: Out-of-range GPU %d specified (%d total GPUs)", device, count);
      return false;
    }
    check_cuda(cudaSetDevice(device));
  }
  ctx->device = device;
  check_cuda(cudaGetDeviceProperties(&dev, device));
  printf("[%d] Using GPU %d: %s\n", ctx->id, device, dev.name);
  return true;
}

void load_graph_CUDA_common(struct CUDA_Context_Common *ctx, MarshalGraph &g, unsigned num_hosts) {
  CSRGraphTy graph;
  ctx->numOwned = g.numOwned;
  ctx->beginMaster = g.beginMaster;
  ctx->numNodesWithEdges = g.numNodesWithEdges;
  assert(ctx->id == g.id);

  size_t mem_usage = ((g.nnodes + 1) + g.nedges) * sizeof(index_type)
    + (g.nnodes) * sizeof(node_data_type);
  if (!g.edge_data) mem_usage += (g.nedges) * sizeof(edge_data_type);
  printf("[%d] Host memory for graph: %3u MB\n", ctx->id, mem_usage/1048756);

  // copy the graph to the GPU
  graph.nnodes = g.nnodes;
  graph.nedges = g.nedges;
  graph.row_start = g.row_start;
  graph.edge_dst = g.edge_dst;
  graph.node_data = g.node_data;
  graph.edge_data = g.edge_data;
  graph.copy_to_gpu(ctx->gg);

  size_t max_shared_size = 0; // for union across master/mirror of all hosts
  ctx->master.num_nodes = (unsigned int *) calloc(num_hosts, sizeof(unsigned int));
  memcpy(ctx->master.num_nodes, g.num_master_nodes, sizeof(unsigned int) * num_hosts);
  ctx->master.nodes = (DeviceOnly<unsigned int> *) calloc(num_hosts, sizeof(Shared<unsigned int>));
  for(uint32_t h = 0; h < num_hosts; ++h){
    if (ctx->master.num_nodes[h] > 0) {
      ctx->master.nodes[h].alloc(ctx->master.num_nodes[h]);
      ctx->master.nodes[h].copy_to_gpu(g.master_nodes[h], ctx->master.num_nodes[h]);
    }
    if (ctx->master.num_nodes[h] > max_shared_size) {
      max_shared_size = ctx->master.num_nodes[h];
    }
  }
  ctx->mirror.num_nodes = (unsigned int *) calloc(num_hosts, sizeof(unsigned int));
  memcpy(ctx->mirror.num_nodes, g.num_mirror_nodes, sizeof(unsigned int) * num_hosts);
  ctx->mirror.nodes = (DeviceOnly<unsigned int> *) calloc(num_hosts, sizeof(Shared<unsigned int>));
  for(uint32_t h = 0; h < num_hosts; ++h){
    if (ctx->mirror.num_nodes[h] > 0) {
      ctx->mirror.nodes[h].alloc(ctx->mirror.num_nodes[h]);
      ctx->mirror.nodes[h].copy_to_gpu(g.mirror_nodes[h], ctx->mirror.num_nodes[h]);
    }
    if (ctx->mirror.num_nodes[h] > max_shared_size) {
      max_shared_size = ctx->mirror.num_nodes[h];
    }
  }
  ctx->offsets.alloc(max_shared_size);
  ctx->is_updated.alloc(1);
  ctx->is_updated.cpu_wr_ptr()->alloc(max_shared_size);
  //printf("[%u] load_graph_GPU: %u owned nodes of total %u resident, %lu edges\n", ctx->id, ctx->nowned, graph.nnodes, graph.nedges);
}

size_t mem_usage_CUDA_common(MarshalGraph &g, unsigned num_hosts) {
  size_t mem_usage = 0;
  size_t max_shared_size = 0; // for union across master/mirror of all hosts
  mem_usage += num_hosts * sizeof(unsigned int);
  mem_usage += num_hosts * sizeof(Shared<unsigned int>);
  for(uint32_t h = 0; h < num_hosts; ++h){
    if (g.num_master_nodes[h] > 0) {
      mem_usage += g.num_master_nodes[h] * sizeof(unsigned int);
    }
    if (g.num_master_nodes[h] > max_shared_size) {
      max_shared_size = g.num_master_nodes[h];
    }
  }
  mem_usage += num_hosts * sizeof(unsigned int);
  mem_usage += num_hosts * sizeof(Shared<unsigned int>);
  for(uint32_t h = 0; h < num_hosts; ++h){
    if (g.num_mirror_nodes[h] > 0) {
      mem_usage += g.num_mirror_nodes[h] * sizeof(unsigned int);
    }
    if (g.num_mirror_nodes[h] > max_shared_size) {
      max_shared_size = g.num_mirror_nodes[h];
    }
  }
  mem_usage += max_shared_size * sizeof(unsigned int);
  mem_usage += ((max_shared_size+63)/64) * sizeof(unsigned long long int);
  return mem_usage;
}

template<typename Type>
void load_graph_CUDA_field(struct CUDA_Context_Common *ctx, struct CUDA_Context_Field<Type> *field, unsigned num_hosts) {
  field->data.alloc(ctx->gg.nnodes);
  size_t max_shared_size = 0; // for union across master/mirror of all hosts
  for(uint32_t h = 0; h < num_hosts; ++h){
    if (ctx->master.num_nodes[h] > max_shared_size) {
      max_shared_size = ctx->master.num_nodes[h];
    }
  }
  for(uint32_t h = 0; h < num_hosts; ++h){
    if (ctx->mirror.num_nodes[h] > max_shared_size) {
      max_shared_size = ctx->mirror.num_nodes[h];
    }
  }
  field->shared_data.alloc(max_shared_size);
  field->is_updated.alloc(1);
  field->is_updated.cpu_wr_ptr()->alloc(ctx->gg.nnodes);
}

template<typename Type>
size_t mem_usage_CUDA_field(struct CUDA_Context_Field<Type> *field, MarshalGraph &g, unsigned num_hosts) {
  size_t mem_usage = 0;
  mem_usage += g.nnodes * sizeof(Type);
  size_t max_shared_size = 0; // for union across master/mirror of all hosts
  for(uint32_t h = 0; h < num_hosts; ++h){
    if (g.num_master_nodes[h] > max_shared_size) {
      max_shared_size = g.num_master_nodes[h];
    }
  }
  for(uint32_t h = 0; h < num_hosts; ++h){
    if (g.num_mirror_nodes[h] > max_shared_size) {
      max_shared_size = g.num_mirror_nodes[h];
    }
  }
  mem_usage += max_shared_size * sizeof(Type);
  mem_usage += ((g.nnodes+63)/64) * sizeof(unsigned long long int);
  return mem_usage;
}
