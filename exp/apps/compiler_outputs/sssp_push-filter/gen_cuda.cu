/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=False $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=1 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=False $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
unsigned int * P_DIST_CURRENT;
unsigned int * P_DIST_OLD;
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
__global__ void InitializeGraph(CSRGraph graph, int  nowned, const unsigned int  local_infinity, unsigned int local_src_node, unsigned int * p_dist_current, unsigned int * p_dist_old)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  // FP: "1 -> 2;
  src_end = nowned;
  for (index_type src = 0 + tid; src < src_end; src += nthreads)
  {
    p_dist_current[src] = (graph.node_data[src] == local_src_node) ? 0 : local_infinity;
    p_dist_old[src] = (graph.node_data[src] == local_src_node) ? 0 : local_infinity;
  }
  // FP: "5 -> 6;
}
__global__ void FirstItr_SSSP(CSRGraph graph, int  nowned, unsigned int * p_dist_current, unsigned int * p_dist_old)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  // FP: "1 -> 2;
  src_end = nowned;
  for (index_type src = 0 + tid; src < src_end; src += nthreads)
  {
    index_type jj_end;
    p_dist_old[src] = p_dist_current[src];
    jj_end = (graph).getFirstEdge((src) + 1);
    for (index_type jj = (graph).getFirstEdge(src) + 0; jj < jj_end; jj += 1)
    {
      index_type dst;
      unsigned int new_dist;
      dst = graph.getAbsDestination(jj);
      new_dist = graph.getAbsWeight(jj) + p_dist_current[src];
      atomicMin(&p_dist_current[dst], new_dist);
    }
  }
  // FP: "11 -> 12;
}
__global__ void SSSP(CSRGraph graph, int  nowned, unsigned int * p_dist_current, unsigned int * p_dist_old, Any any_retval)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  // FP: "1 -> 2;
  src_end = nowned;
  for (index_type src = 0 + tid; src < src_end; src += nthreads)
  {
    if (p_dist_old[src] > p_dist_current[src])
    {
      index_type jj_end;
      p_dist_old[src] = p_dist_current[src];
      any_retval.return_( 1);
      jj_end = (graph).getFirstEdge((src) + 1);
      for (index_type jj = (graph).getFirstEdge(src) + 0; jj < jj_end; jj += 1)
      {
        index_type dst;
        unsigned int new_dist;
        dst = graph.getAbsDestination(jj);
        new_dist = graph.getAbsWeight(jj) + p_dist_current[src];
        atomicMin(&p_dist_current[dst], new_dist);
      }
    }
  }
  // FP: "14 -> 15;
}
void InitializeGraph_cuda(const unsigned int & local_infinity, unsigned int local_src_node, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(ctx->gg, blocks, threads);
  // FP: "4 -> 5;
  InitializeGraph <<<blocks, threads>>>(ctx->gg, ctx->nowned, local_infinity, local_src_node, ctx->dist_current.gpu_wr_ptr(), ctx->dist_old.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void FirstItr_SSSP_cuda(struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(ctx->gg, blocks, threads);
  // FP: "4 -> 5;
  FirstItr_SSSP <<<blocks, threads>>>(ctx->gg, ctx->nowned, ctx->dist_current.gpu_wr_ptr(), ctx->dist_old.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void SSSP_cuda(int & __retval, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(ctx->gg, blocks, threads);
  // FP: "4 -> 5;
  *(ctx->p_retval.cpu_wr_ptr()) = __retval;
  // FP: "5 -> 6;
  ctx->any_retval.rv = ctx->p_retval.gpu_wr_ptr();
  // FP: "6 -> 7;
  SSSP <<<blocks, threads>>>(ctx->gg, ctx->nowned, ctx->dist_current.gpu_wr_ptr(), ctx->dist_old.gpu_wr_ptr(), ctx->any_retval);
  // FP: "7 -> 8;
  check_cuda_kernel;
  // FP: "8 -> 9;
  __retval = *(ctx->p_retval.cpu_rd_ptr());
  // FP: "9 -> 10;
}