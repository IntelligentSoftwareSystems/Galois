/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=1 $ instrument=set([]) $ unroll=[] $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=False $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
unsigned int * P_DIST_CURRENT;
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
__global__ void InitializeGraph(CSRGraph graph, int  nowned, unsigned int local_infinity, int local_src_node, unsigned int * p_dist_current)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  src_end = nowned;
  for (index_type src = 0 + tid; src < src_end; src += nthreads)
  {
    p_dist_current[src] = (src == local_src_node) ? 0 : local_infinity;
  }
}
__global__ void SSSP(CSRGraph graph, int  nowned, unsigned int * p_dist_current, Worklist2 in_wl, Worklist2 out_wl)
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
    unsigned int sdist;
    index_type jj_end;
    pop = (in_wl).pop_id(wlvertex, src);
    sdist = p_dist_current[src];
    jj_end = (graph).getFirstEdge((src) + 1);
    for (index_type jj = (graph).getFirstEdge(src) + 0; jj < jj_end; jj += 1)
    {
      index_type dst;
      unsigned int new_dist;
      unsigned int old_dist;
      dst = graph.getAbsDestination(jj);
      new_dist = graph.getAbsWeight(jj) + sdist;
      old_dist = atomicMin(&p_dist_current[dst], new_dist);
      if (old_dist > new_dist)
      {
        index_type _start_30;
        _start_30 = (out_wl).setup_push_warp_one();;
        (out_wl).do_push(_start_30, 0, dst);
      }
    }
  }
}
void InitializeGraph_cuda(int local_src_node, unsigned int local_infinity, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  kernel_sizing(ctx->gg, blocks, threads);
  InitializeGraph <<<blocks, threads>>>(ctx->gg, ctx->nowned, local_infinity, local_src_node, ctx->dist_current.gpu_wr_ptr());
  check_cuda_kernel;
}
void SSSP_cuda(struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  kernel_sizing(ctx->gg, blocks, threads);
  ctx->in_wl.update_gpu(ctx->shared_wl->num_in_items);
  ctx->out_wl.will_write();
  ctx->out_wl.reset();
  SSSP <<<blocks, threads>>>(ctx->gg, ctx->nowned, ctx->dist_current.gpu_wr_ptr(), ctx->in_wl, ctx->out_wl);
  check_cuda_kernel;
  ctx->out_wl.update_cpu();
  ctx->shared_wl->num_out_items = ctx->out_wl.nitems();
}