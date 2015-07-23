/*  -*- mode: c++ -*-  */
#ifdef BACKEND_GPU_CUDA
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraphTex &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "unroll=set([]) $ ";
struct PRNode { float* pr_curr; float* pr_next; unsigned int* nout; };
__global__ void initialize_node(PRNode node_data, CSRGraphTex graph)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  index_type node_end;
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    index_type e_end;
    node_data.pr_curr[node] = (1.0) - (ALPHA);
    node_data.pr_next[node] = 0.0;
    e_end = (graph).getFirstEdge((node) + 1);
    for (index_type e = (graph).getFirstEdge(node) + 0; e < e_end; e += 1)
    {
      index_type dn;
      unsigned int x;
      dn = graph.getAbsDestination(e);
      x = (hgalois$Atomics).add(node_data.nout[dn],1);
    }
  }
}
__global__ void pr(PRNode node_data, CSRGraphTex graph)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  index_type node_end;
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    float value;
    index_type e_end;
    value = 0.0;
    e_end = (graph).getFirstEdge((node) + 1);
    for (index_type e = (graph).getFirstEdge(node) + 0; e < e_end; e += 1)
    {
      index_type dn;
      dn = graph.getAbsDestination(e);
      if (node_data.nout[dn])
      {
        value += (node_data.pr_curr[dn]) / (node_data.nout[dn]);
      }
    }
    node_data.pr_next[node] = (((1.0) - (ALPHA)) * (value)) + (ALPHA);
  }
}
__global__ void output(PRNode node_data)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  index_type node_end;
  node_end = (None).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
  }
}
unsigned int get_PRNode_nout_CUDA(struct pr_CUDA_Context *ctx, unsigned LID) {
}

void set_PRNode_nout_CUDA(struct pr_CUDA_Context *ctx, unsigned LID, unsigned int nout) {
}

void set_PRNode_nout_plus_CUDA(struct pr_CUDA_Context *ctx, unsigned LID, unsigned int nout) {
}

float get_PRNode_pr_CUDA(struct pr_CUDA_Context *ctx, unsigned LID) {
}

void set_PRNode_pr_CUDA(struct pr_CUDA_Context *ctx, unsigned LID, float pr) {
}

#endif