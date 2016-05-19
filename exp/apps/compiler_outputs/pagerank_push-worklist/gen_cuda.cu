/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=True $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['wp', 'fg']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
unsigned int * P_NOUT;
float * P_RESIDUAL;
float * P_VALUE;
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
__global__ void InitializeGraph(CSRGraph graph, int  nowned, const float  local_alpha, unsigned int * p_nout, float * p_residual, float * p_value)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  src_end = nowned;
  for (index_type src = 0 + tid; src < src_end; src += nthreads)
  {
    p_value[src] = local_alpha;
    p_nout[src] = graph.getOutDegree(src);
    if (p_nout[src] > 0)
    {
      float delta;
      index_type nbr_end;
      delta = p_value[src]*(1-local_alpha)/p_nout[src];
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
__global__ void PageRank(CSRGraph graph, int  nowned, const float  local_alpha, float local_tolerance, unsigned int * p_nout, float * p_residual, float * p_value, Worklist2 in_wl, Worklist2 out_wl)
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
      delta = residual_old*(1-local_alpha)/p_nout[src];
      nbr_end = (graph).getFirstEdge((src) + 1);
      for (index_type nbr = (graph).getFirstEdge(src) + 0; nbr < nbr_end; nbr += 1)
      {
        index_type dst;
        float dst_residual_old;
        dst = graph.getAbsDestination(nbr);
        dst_residual_old = atomicAdd(&p_residual[dst], delta);
        if ((dst_residual_old <= local_tolerance) && ((dst_residual_old + delta) >= local_tolerance))
        {
          index_type _start_46;
          _start_46 = (out_wl).setup_push_warp_one();;
          (out_wl).do_push(_start_46, 0, dst);
        }
      }
    }
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
  kernel_sizing(ctx->gg, blocks, threads);
  ctx->in_wl.update_gpu(ctx->shared_wl->num_in_items);
  ctx->out_wl.will_write();
  ctx->out_wl.reset();
  PageRank <<<blocks, threads>>>(ctx->gg, ctx->nowned, local_alpha, local_tolerance, ctx->nout.gpu_wr_ptr(), ctx->residual.gpu_wr_ptr(), ctx->value.gpu_wr_ptr(), ctx->in_wl, ctx->out_wl);
  check_cuda_kernel;
  ctx->out_wl.update_cpu();
  ctx->shared_wl->num_out_items = ctx->out_wl.nitems();
}
