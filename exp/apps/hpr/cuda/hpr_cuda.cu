/* -*- mode: c++ -*- */
#include <cuda.h>
#include <stdio.h>
#include "gg.h"
#include "ggcuda.h"
#include "hpr_cuda.h"
#include "../hpr.h"

struct CUDA_Context {
  size_t nowned;
  size_t g_offset;
  CSRGraphTy hg;
  CSRGraphTy gg;
  Shared<float> pr[2];
  Shared<int> nout;
};

__global__ void test_cuda_too(CSRGraphTy g) {
  int tid = TID_1D;
  int num_threads = TOTAL_THREADS_1D;

  for(int i = tid; i < g.nnodes; i+=num_threads) {
    printf("%d %d %d\n", tid, i, g.getOutDegree(i));
  }
}

__global__ void initialize_graph(CSRGraphTy g, index_type nowned, 
				 float *pcur, float *pnext, int *nout) {
  int tid = TID_1D;
  int num_threads = TOTAL_THREADS_1D;

  for(int i = tid; i < nowned; i += num_threads) {
    pcur[i] = 1.0 - alpha;
    pnext[i] = 0;
    nout[i] = g.getOutDegree(i);
  }
}

struct CUDA_Context *get_CUDA_context() {
  return (struct CUDA_Context *) calloc(1, sizeof(struct CUDA_Context));
}

void setNodeValue_CUDA(struct CUDA_Context *ctx, unsigned LID, float v) {
  float *pr = ctx->pr[0].cpu_wr_ptr();
  
  pr[LID] = v;
}

void setNodeAttr_CUDA(struct CUDA_Context *ctx, unsigned LID, unsigned nout) {
  int *pnout = ctx->nout.cpu_wr_ptr();
  
  pnout[LID] = nout;
}

void load_graph_CUDA(struct CUDA_Context *ctx, MarshalGraph &g) {
  CSRGraphTy &graph = ctx->hg;

  ctx->nowned = g.nowned;
  
  graph.nnodes = g.nnodes;
  graph.nedges = g.nedges;

  if(!graph.allocOnHost()) {
    fprintf(stderr, "Unable to alloc space for graph!");
    exit(1);
  }
  
  memcpy(graph.row_start, g.row_start, sizeof(index_type) * (g.nnodes + 1));
  memcpy(graph.edge_dst, g.edge_dst, sizeof(index_type) * g.nedges);

  if(g.node_data)
    memcpy(graph.node_data, g.node_data, sizeof(node_data_type) * g.nnodes);

  if(g.edge_data)
    memcpy(graph.edge_data, g.edge_data, sizeof(edge_data_type) * g.nedges);

  graph.copy_to_gpu(ctx->gg);

  ctx->pr[0].alloc(graph.nnodes);
  ctx->pr[1].alloc(graph.nnodes);
  ctx->nout.alloc(graph.nnodes);

  printf("load_graph_GPU: %d owned nodes of total %d resident\n", 
	 ctx->nowned, graph.nnodes);  
}

void initialize_graph_cuda(struct CUDA_Context *ctx) {  
  initialize_graph<<<14, 256>>>(ctx->gg, ctx->nowned, ctx->pr[0].gpu_wr_ptr(), ctx->pr[1].gpu_wr_ptr(), ctx->nout.gpu_wr_ptr());
  check_cuda(cudaDeviceSynchronize());
}

void test_cuda(struct CUDA_Context *ctx) {
  printf("hello from cuda!\n");
  CSRGraphTy &gg = ctx->gg;

  test_cuda_too<<<1, 1>>>(gg);
  check_cuda(cudaDeviceSynchronize());
}
