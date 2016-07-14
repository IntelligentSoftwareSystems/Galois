/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=False $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=1 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=False $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
int * P_NOUT;
float * P_VALUE;
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
__global__ void ResetGraph(CSRGraph graph, int  nowned, int * p_nout, float * p_value)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  // FP: "1 -> 2;
  src_end = nowned;
  for (index_type src = 0 + tid; src < src_end; src += nthreads)
  {
    p_value[src] = 0;
    p_nout[src] = 0;
  }
  // FP: "5 -> 6;
}
__global__ void InitializeGraph(CSRGraph graph, int  nowned, const float  local_alpha, int * p_nout, float * p_value)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  // FP: "1 -> 2;
  src_end = nowned;
  for (index_type src = 0 + tid; src < src_end; src += nthreads)
  {
    index_type nbr_end;
    p_value[src] = local_alpha;
    nbr_end = (graph).getFirstEdge((src) + 1);
    for (index_type nbr = (graph).getFirstEdge(src) + 0; nbr < nbr_end; nbr += 1)
    {
      index_type dst;
      dst = graph.getAbsDestination(nbr);
      atomicAdd(&p_nout[dst], 1);
    }
  }
  // FP: "9 -> 10;
}
__global__ void PageRank(CSRGraph graph, int  nowned, const float  local_alpha, float local_tolerance, int * p_nout, float * p_value, Any any_retval)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  // FP: "1 -> 2;
  src_end = nowned;
  for (index_type src = 0 + tid; src < src_end; src += nthreads)
  {
    float sum;
    index_type nbr_end;
    float pr_value;
    float diff;
    sum = 0;
    nbr_end = (graph).getFirstEdge((src) + 1);
    for (index_type nbr = (graph).getFirstEdge(src) + 0; nbr < nbr_end; nbr += 1)
    {
      index_type dst;
      unsigned int dnout;
      dst = graph.getAbsDestination(nbr);
      dnout = p_nout[dst];
      if (dnout > 0)
      {
        sum += p_value[dst]/dnout;
      }
    }
    pr_value = sum*(1.0 - local_alpha) + local_alpha;
    diff = fabs(pr_value - p_value[src]);
    if (diff > local_tolerance)
    {
      p_value[src] = pr_value;
      any_retval.return_( 1);
    }
  }
  // FP: "22 -> 23;
}
void ResetGraph_cuda(struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(ctx->gg, blocks, threads);
  // FP: "4 -> 5;
  ResetGraph <<<blocks, threads>>>(ctx->gg, ctx->nowned, ctx->nout.gpu_wr_ptr(), ctx->value.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void InitializeGraph_cuda(const float & local_alpha, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(ctx->gg, blocks, threads);
  // FP: "4 -> 5;
  InitializeGraph <<<blocks, threads>>>(ctx->gg, ctx->nowned, local_alpha, ctx->nout.gpu_wr_ptr(), ctx->value.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void PageRank_cuda(int & __retval, const float & local_alpha, float local_tolerance, struct CUDA_Context * ctx)
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
  PageRank <<<blocks, threads>>>(ctx->gg, ctx->nowned, local_alpha, local_tolerance, ctx->nout.gpu_wr_ptr(), ctx->value.gpu_wr_ptr(), ctx->any_retval);
  // FP: "7 -> 8;
  check_cuda_kernel;
  // FP: "8 -> 9;
  __retval = *(ctx->p_retval.cpu_rd_ptr());
  // FP: "9 -> 10;
}