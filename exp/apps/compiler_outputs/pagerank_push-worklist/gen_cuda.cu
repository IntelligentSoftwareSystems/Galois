/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['wp', 'fg']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
unsigned int * P_NOUT;
float * P_RESIDUAL;
float * P_VALUE;
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
static const int __tb_PageRank = TB_SIZE;
__global__ void ResetGraph(CSRGraph graph, int  nowned, unsigned int * p_nout, float * p_residual, float * p_value)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  src_end = nowned;
  for (index_type src = 0 + tid; src < src_end; src += nthreads)
  {
    p_value[src] = 0;
    p_nout[src] = 0;
    p_residual[src] = 0;
  }
}
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

  const unsigned __kernel_tb_size = __tb_PageRank;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type wlvertex_end;
  const int _NP_CROSSOVER_WP = 32;
  const int _NP_CROSSOVER_TB = __kernel_tb_size;
  const int BLKSIZE = __kernel_tb_size;
  const int ITSIZE = BLKSIZE * 8;

  typedef cub::BlockScan<multiple_sum<2, index_type>, BLKSIZE> BlockScan;
  typedef union np_shared<BlockScan::TempStorage, index_type, struct empty_np, struct warp_np<__kernel_tb_size/32>, struct fg_np<ITSIZE> > npsTy;

  __shared__ npsTy nps ;
  wlvertex_end = roundup((*((volatile index_type *) (in_wl).dindex)), (blockDim.x));
  for (index_type wlvertex = 0 + tid; wlvertex < wlvertex_end; wlvertex += nthreads)
  {
    int src;
    bool pop;
    float residual_old;
    float delta;
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    pop = (in_wl).pop_id(wlvertex, src);
    if (pop)
    {
      residual_old = atomicExch(&p_residual[src], 0.0);
      p_value[src] += residual_old;
      delta = 0;
      if (p_nout[src] > 0)
      {
        delta = residual_old*(1-local_alpha)/p_nout[src];
      }
    }
    struct NPInspector1 _np = {0,0,0,0,0,0};
    __shared__ struct { float delta; } _np_closure [TB_SIZE];
    _np_closure[threadIdx.x].delta = delta;
    if (pop)
    {
      _np.size = (graph).getOutDegree(src);
      _np.start = (graph).getFirstEdge(src);
    }
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    if (threadIdx.x == 0)
    {
    }
    __syncthreads();
    {
      const int warpid = threadIdx.x / 32;
      const int _np_laneid = cub::LaneId();
      while (__any(_np.size >= _NP_CROSSOVER_WP))
      {
        if (_np.size >= _NP_CROSSOVER_WP)
        {
          nps.warp.owner[warpid] = _np_laneid;
        }
        if (nps.warp.owner[warpid] == _np_laneid)
        {
          nps.warp.start[warpid] = _np.start;
          nps.warp.size[warpid] = _np.size;
          nps.warp.src[warpid] = threadIdx.x;
          _np.start = 0;
          _np.size = 0;
        }
        index_type _np_w_start = nps.warp.start[warpid];
        index_type _np_w_size = nps.warp.size[warpid];
        assert(nps.warp.src[warpid] < __kernel_tb_size);
        delta = _np_closure[nps.warp.src[warpid]].delta;
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type nbr;
          nbr = _np_w_start +_np_ii;
          {
            index_type dst;
            float dst_residual_old;
            dst = graph.getAbsDestination(nbr);
            dst_residual_old = atomicAdd(&p_residual[dst], delta);
            if ((dst_residual_old <= local_tolerance) && ((dst_residual_old + delta) >= local_tolerance))
            {
              index_type _start_58;
              _start_58 = (out_wl).setup_push_warp_one();;
              (out_wl).do_push(_start_58, 0, dst);
            }
          }
        }
      }
      __syncthreads();
    }

    __syncthreads();
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    while (_np.work())
    {
      int _np_i =0;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      __syncthreads();

      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type nbr;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        delta = _np_closure[nps.fg.src[_np_i]].delta;
        nbr= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          float dst_residual_old;
          dst = graph.getAbsDestination(nbr);
          dst_residual_old = atomicAdd(&p_residual[dst], delta);
          if ((dst_residual_old <= local_tolerance) && ((dst_residual_old + delta) >= local_tolerance))
          {
            index_type _start_58;
            _start_58 = (out_wl).setup_push_warp_one();;
            (out_wl).do_push(_start_58, 0, dst);
          }
        }
      }
      _np.execute_round_done(ITSIZE);
      __syncthreads();
    }
    assert(threadIdx.x < __kernel_tb_size);
    delta = _np_closure[threadIdx.x].delta;
  }
}
void ResetGraph_cuda(struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  kernel_sizing(ctx->gg, blocks, threads);
  ResetGraph <<<blocks, threads>>>(ctx->gg, ctx->nowned, ctx->nout.gpu_wr_ptr(), ctx->residual.gpu_wr_ptr(), ctx->value.gpu_wr_ptr());
  check_cuda_kernel;
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
  PageRank <<<blocks, __tb_PageRank>>>(ctx->gg, ctx->nowned, local_alpha, local_tolerance, ctx->nout.gpu_wr_ptr(), ctx->residual.gpu_wr_ptr(), ctx->value.gpu_wr_ptr(), ctx->in_wl, ctx->out_wl);
  check_cuda_kernel;
  ctx->out_wl.update_cpu();
  ctx->shared_wl->num_out_items = ctx->out_wl.nitems();
}