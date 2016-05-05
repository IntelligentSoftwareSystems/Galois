/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraphTex &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=False $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=1 $ instrument=set([]) $ unroll=[] $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=False $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=texture $ cuda.use_worklist_slots=True $ cuda.worklist_type=texture";
int * P_NOUT;
float * P_VALUE;
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
__global__ void InitializeGraph(CSRGraphTex graph, int  nowned, const float  local_alpha, int * p_nout, float * p_value)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  src_end = nowned;
  for (index_type src = 0 + tid; src < src_end; src += nthreads)
  {
    index_type nbr_end;
    p_value[src] = 1.0 - local_alpha;
    nbr_end = (graph).getFirstEdge((src) + 1);
    for (index_type nbr = (graph).getFirstEdge(src) + 0; nbr < nbr_end; nbr += 1)
    {
      index_type dst;
      dst = graph.getAbsDestination(nbr);
      atomicAdd(&p_nout[dst], 1);
    }
  }
}
__global__ void PageRank_pull(CSRGraphTex graph, int  nowned, const float  local_alpha, float local_tolerance, int * p_nout, float * p_value, WorklistT in_wl, WorklistT out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type wlvertex_end;
  wlvertex_end = *((volatile index_type *) (in_wl).dindex);
  for (index_type wlvertex = 0 + tid; wlvertex < wlvertex_end; wlvertex += nthreads)
  {
    int src;
    bool pop;
    float sum;
    index_type nbr_end;
    float pr_value;
    float diff;
    pop = (in_wl).pop_id(wlvertex, src);
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
      (out_wl).push(src);
    }
  }
}
void InitializeGraph_cuda(const float & local_alpha, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  kernel_sizing(ctx->gg, blocks, threads);
  InitializeGraph <<<blocks, threads>>>(ctx->gg, ctx->nowned, local_alpha, ctx->nout.gpu_wr_ptr(), ctx->value.gpu_wr_ptr());
  check_cuda_kernel;
}
void PageRank_pull_cuda(const float & local_alpha, float local_tolerance, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  kernel_sizing(ctx->gg, blocks, threads);
  ctx->in_wl.update_gpu(ctx->shared_wl->num_in_items);
  ctx->out_wl.will_write();
  ctx->out_wl.reset();
  PageRank_pull <<<blocks, threads>>>(ctx->gg, ctx->nowned, local_alpha, local_tolerance, ctx->nout.gpu_wr_ptr(), ctx->value.gpu_wr_ptr(), ctx->in_wl, ctx->out_wl);
  check_cuda_kernel;
  ctx->out_wl.update_cpu();
  ctx->shared_wl->num_out_items = ctx->out_wl.nitems();
}