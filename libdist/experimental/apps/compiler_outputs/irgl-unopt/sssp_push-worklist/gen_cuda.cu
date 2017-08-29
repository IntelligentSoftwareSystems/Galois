/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=1 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=False $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
unsigned int * P_DIST_CURRENT;
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
__global__ void InitializeGraph(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, const unsigned int  local_infinity, unsigned int local_src_node, unsigned int * p_dist_current)
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
      p_dist_current[src] = (graph.node_data[src] == local_src_node) ? 0 : local_infinity;
    }
  }
  // FP: "7 -> 8;
}
__global__ void SSSP(CSRGraph graph, unsigned int __nowned, unsigned int * p_dist_current, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type wlvertex_end;
  // FP: "1 -> 2;
  wlvertex_end = *((volatile index_type *) (in_wl).dindex);
  for (index_type wlvertex = 0 + tid; wlvertex < wlvertex_end; wlvertex += nthreads)
  {
    int src;
    bool pop;
    index_type jj_end;
    pop = (in_wl).pop_id(wlvertex, src);
    if (pop)
    {
    }
    if (!pop)
    {
      continue;
    }
    jj_end = (graph).getFirstEdge((src) + 1);
    for (index_type jj = (graph).getFirstEdge(src) + 0; jj < jj_end; jj += 1)
    {
      index_type dst;
      unsigned int new_dist;
      unsigned int old_dist;
      dst = graph.getAbsDestination(jj);
      new_dist = graph.getAbsWeight(jj) + p_dist_current[src];
      old_dist = atomicMin(&p_dist_current[dst], new_dist);
      if (old_dist > new_dist)
      {
        index_type _start_39;
        _start_39 = (out_wl).setup_push_warp_one();;
        (out_wl).do_push(_start_39, 0, dst);
      }
    }
  }
  // FP: "24 -> 25;
}
void InitializeGraph_cuda(unsigned int  __begin, unsigned int  __end, const unsigned int & local_infinity, unsigned int local_src_node, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  InitializeGraph <<<blocks, threads>>>(ctx->gg, ctx->nowned, __begin, __end, local_infinity, local_src_node, ctx->dist_current.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void InitializeGraph_all_cuda(const unsigned int & local_infinity, unsigned int local_src_node, struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  InitializeGraph_cuda(0, ctx->nowned, local_infinity, local_src_node, ctx);
  // FP: "2 -> 3;
}
void SSSP_cuda(struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  ctx->in_wl.update_gpu(ctx->shared_wl->num_in_items);
  // FP: "5 -> 6;
  ctx->out_wl.will_write();
  // FP: "6 -> 7;
  ctx->out_wl.reset();
  // FP: "7 -> 8;
  SSSP <<<blocks, threads>>>(ctx->gg, ctx->nowned, ctx->dist_current.gpu_wr_ptr(), ctx->in_wl, ctx->out_wl);
  // FP: "8 -> 9;
  check_cuda_kernel;
  // FP: "9 -> 10;
  ctx->out_wl.update_cpu();
  // FP: "10 -> 11;
  ctx->shared_wl->num_out_items = ctx->out_wl.nitems();
  // FP: "11 -> 12;
}