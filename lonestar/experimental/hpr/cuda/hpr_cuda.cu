/* -*- mode: c++ -*- */
#include <cuda.h>
#include <stdio.h>
#include "gg.h"
#include "ggcuda.h"
#include "hpr_cuda.h"
#include "../hpr.h"

struct CUDA_Context {
  int device;
  int id;
  int pr_it;
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
  }
}


__global__ void initialize_nout(CSRGraphTy graph, index_type nowned, int *nout) {
  int tid = TID_1D;
  int num_threads = TOTAL_THREADS_1D;

  for(int i = tid; i < graph.nnodes; i += num_threads) {
    index_type edge_end = graph.getFirstEdge(i + 1);

    for(index_type e = graph.getFirstEdge(i); e < edge_end; e++) {
      index_type dst = graph.getAbsDestination(e);
      assert(dst < graph.nnodes);
      if(!(i >= nowned && dst >= nowned))
	atomicAdd(nout + dst, 1);
    }
  }
}


__global__ void pagerank(CSRGraphTy graph, index_type nowned, 
			 float *pr, float *pr_next, int *nout) {
  int tid = TID_1D;
  int num_threads = TOTAL_THREADS_1D;

  for(int i = tid; i < nowned; i += num_threads) {
    float sum = 0.0;

    index_type edge_end = graph.getFirstEdge(i + 1);

    for(index_type e = graph.getFirstEdge(i); e < edge_end; e++) {
      index_type dst = graph.getAbsDestination(e);

      if(nout[dst] != 0)  // can nout == 0?
	sum += pr[dst] / nout[dst];      
      else 
	printf("WARNING: %d %d is zero\n", i, dst);
    }

    float value = (1.0 - alpha) * sum + alpha;
    float diff = fabs(value - pr[i]);
    pr_next[i] = value;
  }
}

__global__ void test_graph(CSRGraphTy g, index_type nowned, 
			   float *pcur, float *pnext, int *nout, int id) {
  int tid = TID_1D;
  int num_threads = TOTAL_THREADS_1D;

  for(int i = tid; i < g.nnodes; i += num_threads) {
    printf("%d %d %d %d\n", id, tid, i, nout[i]); // i is LID!
  }
}


struct CUDA_Context *get_CUDA_context(int id) {
  struct CUDA_Context *p;
  p = (struct CUDA_Context *) calloc(1, sizeof(struct CUDA_Context));
  p->id = id;
  p->pr_it = 0;
  return p;
}

bool init_CUDA_context(struct CUDA_Context *ctx, int device) {
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

  fprintf(stderr, "%d: Using GPU %d: %s\n", ctx->id, device, dev.name);
  return true;
}

float getNodeValue_CUDA(struct CUDA_Context *ctx, unsigned LID) {
  float *pr = ctx->pr[ctx->pr_it].cpu_rd_ptr();

  return pr[LID];
}

void setNodeValue_CUDA(struct CUDA_Context *ctx, unsigned LID, float v) {
  float *pr = ctx->pr[ctx->pr_it].cpu_wr_ptr();
  
  pr[LID] = v;
}

void setNodeAttr_CUDA(struct CUDA_Context *ctx, unsigned LID, unsigned nout) {
  int *pnout = ctx->nout.cpu_wr_ptr();

  assert(LID >= ctx->nowned);
  
//  printf("setting %d %d %d\n", ctx->id, LID, nout);
  pnout[LID] = nout;
}

void setNodeAttr2_CUDA(struct CUDA_Context *ctx, unsigned LID, unsigned nout) {
  int *pnout = ctx->nout.cpu_wr_ptr();

  //printf("setting %d %d %d\n", ctx->id, LID, nout);

  assert(LID < ctx->nowned);
  
  pnout[LID] += nout;
}

unsigned getNodeAttr_CUDA(struct CUDA_Context *ctx, unsigned LID) {
  int *pnout = ctx->nout.cpu_rd_ptr();
  
  assert(LID < ctx->nowned);

  return pnout[LID];
}

unsigned getNodeAttr2_CUDA(struct CUDA_Context *ctx, unsigned LID) {
  int *pnout = ctx->nout.cpu_rd_ptr();
  
  assert(LID >= ctx->nowned);

  return pnout[LID];
}

void load_graph_CUDA(struct CUDA_Context *ctx, MarshalGraph &g) {
  CSRGraphTy &graph = ctx->hg;

  ctx->nowned = g.nowned;
  ctx->id = g.id;
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

  printf("load_graph_GPU: %d owned nodes of total %d resident, %d edges\n", 
	 ctx->nowned, graph.nnodes, graph.nedges);  
}

void initialize_graph_cuda(struct CUDA_Context *ctx) {  
  ctx->nout.zero_gpu();
  initialize_graph<<<14, 256>>>(ctx->gg, ctx->nowned, ctx->pr[0].gpu_wr_ptr(), ctx->pr[1].gpu_wr_ptr(), ctx->nout.gpu_wr_ptr());

  initialize_nout<<<14, 256>>>(ctx->gg, ctx->nowned, ctx->nout.gpu_wr_ptr());
  check_cuda(cudaDeviceSynchronize());
}

void pagerank_cuda(struct CUDA_Context *ctx) {  
  pagerank<<<14, 256>>>(ctx->gg, ctx->nowned, 
			ctx->pr[ctx->pr_it].gpu_wr_ptr(), 
			ctx->pr[ctx->pr_it ^ 1].gpu_wr_ptr(), 
			ctx->nout.gpu_wr_ptr());
  ctx->pr_it ^= 1;  // not sure this is to be done here
  check_cuda(cudaDeviceSynchronize());
}

void test_graph_cuda(struct CUDA_Context *ctx) {  
  test_graph<<<14, 256>>>(ctx->gg, ctx->nowned, ctx->pr[0].gpu_wr_ptr(), ctx->pr[1].gpu_wr_ptr(), ctx->nout.gpu_wr_ptr(), ctx->id);
  check_cuda(cudaDeviceSynchronize());
}


void test_cuda(struct CUDA_Context *ctx) {
  printf("hello from cuda!\n");
  CSRGraphTy &gg = ctx->gg;

  test_cuda_too<<<1, 1>>>(gg);
  check_cuda(cudaDeviceSynchronize());
}
