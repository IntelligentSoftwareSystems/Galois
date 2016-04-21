/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraphTex &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=False $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=1 $ instrument=set([]) $ unroll=[] $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=False $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=texture $ cuda.use_worklist_slots=True $ cuda.worklist_type=texture";
unsigned int * P_NOUT;
float * P_RESIDUAL;
float * P_VALUE;
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
__global__ void InitializeGraph(CSRGraphTex graph, int  nowned, const float  local_alpha, unsigned int * p_nout, float * p_residual, float * p_value)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  src_end = nowned;
  for (index_type src = 0 + tid; src < src_end; src += nthreads)
  {
    p_value[src] = 1.0 - local_alpha;
    p_nout[src] = graph.getOutDegree(src);
    if (p_nout[src] > 0)
    {
      float delta;
      index_type nbr_end;
      delta = p_value[src]*local_alpha/p_nout[src];
      nbr_end = (graph).getFirstEdge((src) + 1);
      for (index_type nbr = (graph).getFirstEdge(src) + 0; nbr < nbr_end; nbr += 1)
      {
        index_type dst;
        dst = graph.getAbsDestination(nbr);
        atomicAdd(&p_residual[dst], delta);
      }
    }
  }
}
__global__ void PageRank(CSRGraphTex graph, int  nowned, const float  local_alpha, float local_tolerance, unsigned int * p_nout, float * p_residual, float * p_value, WorklistT in_wl, WorklistT out_wl)
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
    float residual_old;
    pop = (in_wl).pop_id(wlvertex, src);
    residual_old = atomicExch(&p_residual[src], 0.0);
    p_value[src] += residual_old;
    if (p_nout[src] > 0)
    {
      float delta;
      index_type nbr_end;
      delta = residual_old*local_alpha/p_nout[src];
      nbr_end = (graph).getFirstEdge((src) + 1);
      for (index_type nbr = (graph).getFirstEdge(src) + 0; nbr < nbr_end; nbr += 1)
      {
        index_type dst;
        float dst_residual_old;
        dst = graph.getAbsDestination(nbr);
        dst_residual_old = atomicAdd(&p_residual[dst], delta);
        if ((dst_residual_old <= local_tolerance) && ((dst_residual_old + delta) >= local_tolerance))
        {
          (out_wl).push(dst);
        }
      }
    }
  }
}
__global__ void __init_worklist__(CSRGraphTex graph, WorklistT in_wl, WorklistT out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type vertex_end;
  vertex_end = (graph).nnodes;
  for (index_type vertex = 0 + tid; vertex < vertex_end; vertex += nthreads)
  {
    (out_wl).push(vertex);
  }
}
void InitializeGraph_cuda(const float & local_alpha, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  kernel_sizing(ctx->gg, blocks, threads);
  InitializeGraph <<<blocks, threads>>>(ctx->gg, ctx->nowned, local_alpha, ctx->nout.gpu_wr_ptr(), ctx->residual.gpu_wr_ptr(), ctx->value.gpu_wr_ptr());
  check_cuda_kernel;
}
void PageRank_cuda(const float & local_alpha, float local_tolerance, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  PipeContextT<WorklistT> pipe;
  kernel_sizing(ctx->gg, blocks, threads);
  pipe = PipeContextT<WorklistT>(ctx->hg.nedges);
  {
    {
      pipe.out_wl().will_write();
      __init_worklist__ <<<blocks, threads>>>(ctx->gg, pipe.in_wl(), pipe.out_wl());
      pipe.in_wl().swap_slots();
      pipe.advance2();
      check_cuda_kernel;
      while (pipe.in_wl().nitems())
      {
        pipe.out_wl().will_write();
        PageRank <<<blocks, threads>>>(ctx->gg, ctx->nowned, local_alpha, local_tolerance, ctx->nout.gpu_wr_ptr(), ctx->residual.gpu_wr_ptr(), ctx->value.gpu_wr_ptr(), pipe.in_wl(), pipe.out_wl());
        pipe.in_wl().swap_slots();
        pipe.advance2();
        check_cuda_kernel;
      }
    }
  }
}