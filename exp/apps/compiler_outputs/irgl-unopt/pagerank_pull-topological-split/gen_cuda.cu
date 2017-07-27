/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=1 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=False $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
int * P_NOUT;
float * P_SUM;
float * P_VALUE;
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
__global__ void ResetGraph(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, int * p_nout, float * p_value)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  // FP: "1 -> 2;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      p_value[src] = 0;
      p_nout[src] = 0;
    }
  }
  // FP: "8 -> 9;
}
__global__ void InitializeGraph(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, const float  local_alpha, int * p_nout, float * p_value)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  // FP: "1 -> 2;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    index_type nbr_end;
    bool pop  = src < __end;
    if (pop)
    {
      p_value[src] = local_alpha;
    }
    if (!pop)
    {
      continue;
    }
    nbr_end = (graph).getFirstEdge((src) + 1);
    for (index_type nbr = (graph).getFirstEdge(src) + 0; nbr < nbr_end; nbr += 1)
    {
      index_type dst;
      dst = graph.getAbsDestination(nbr);
      atomicAdd(&p_nout[dst], 1);
    }
  }
  // FP: "15 -> 16;
}
__global__ void PageRank_partial(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, int * p_nout, float * p_sum, float * p_value)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  // FP: "1 -> 2;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    index_type nbr_end;
    bool pop  = src < __end;
    if (pop)
    {
      p_sum[src] = 0;
    }
    if (!pop)
    {
      continue;
    }
    nbr_end = (graph).getFirstEdge((src) + 1);
    for (index_type nbr = (graph).getFirstEdge(src) + 0; nbr < nbr_end; nbr += 1)
    {
      index_type dst;
      unsigned int dnout;
      dst = graph.getAbsDestination(nbr);
      dnout = p_nout[dst];
      if (dnout > 0)
      {
        atomicAdd(&p_sum[src], p_value[dst]/dnout);
      }
    }
  }
  // FP: "19 -> 20;
}
__global__ void PageRank(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, const float  local_alpha, float local_tolerance, float * p_sum, float * p_value, Sum ret_val)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  typedef cub::BlockReduce<int, TB_SIZE> _br;
  __shared__ _br::TempStorage _ts;
  ret_val.thread_entry();
  float pr_value;
  float diff;
  index_type src_end;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      pr_value = p_sum[src]*(1.0 - local_alpha) + local_alpha;
      diff = pr_value - p_value[src];
      if (diff > local_tolerance)
      {
        p_value[src] = pr_value;
        ret_val.do_return( 1);
        continue;
      }
    }
  }
  ret_val.thread_exit<_br>(_ts);
}
void ResetGraph_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  ResetGraph <<<blocks, threads>>>(ctx->gg, ctx->nowned, __begin, __end, ctx->nout.gpu_wr_ptr(), ctx->value.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void ResetGraph_all_cuda(struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  ResetGraph_cuda(0, ctx->nowned, ctx);
  // FP: "2 -> 3;
}
void InitializeGraph_cuda(unsigned int  __begin, unsigned int  __end, const float & local_alpha, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  InitializeGraph <<<blocks, threads>>>(ctx->gg, ctx->nowned, __begin, __end, local_alpha, ctx->nout.gpu_wr_ptr(), ctx->value.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void InitializeGraph_all_cuda(const float & local_alpha, struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  InitializeGraph_cuda(0, ctx->nowned, local_alpha, ctx);
  // FP: "2 -> 3;
}
void PageRank_partial_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  PageRank_partial <<<blocks, threads>>>(ctx->gg, ctx->nowned, __begin, __end, ctx->nout.gpu_wr_ptr(), ctx->sum.gpu_wr_ptr(), ctx->value.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void PageRank_partial_all_cuda(struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  PageRank_partial_cuda(0, ctx->nowned, ctx);
  // FP: "2 -> 3;
}
void PageRank_cuda(unsigned int  __begin, unsigned int  __end, int & __retval, const float & local_alpha, float local_tolerance, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  Shared<int> retval = Shared<int>(1);
  Sum _rv;
  *(retval.cpu_wr_ptr()) = 0;
  _rv.rv = retval.gpu_wr_ptr();
  PageRank <<<blocks, threads>>>(ctx->gg, ctx->nowned, __begin, __end, local_alpha, local_tolerance, ctx->sum.gpu_wr_ptr(), ctx->value.gpu_wr_ptr(), _rv);
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
  __retval = *(retval.cpu_rd_ptr());
  // FP: "7 -> 8;
}
void PageRank_all_cuda(int & __retval, const float & local_alpha, float local_tolerance, struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  PageRank_cuda(0, ctx->nowned, __retval, local_alpha, local_tolerance, ctx);
  // FP: "2 -> 3;
}
