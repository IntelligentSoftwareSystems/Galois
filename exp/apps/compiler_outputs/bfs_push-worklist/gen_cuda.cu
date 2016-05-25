/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=True $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
unsigned int * P_DIST_CURRENT;
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
static const int __tb_BFS = TB_SIZE;
__global__ void InitializeGraph(CSRGraph graph, int  nowned, unsigned int local_infinity, unsigned int local_src_node, unsigned int * p_dist_current)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  src_end = nowned;
  for (index_type src = 0 + tid; src < src_end; src += nthreads)
  {
    p_dist_current[src] = (graph.node_data[src] == local_src_node) ? 0 : local_infinity;
  }
}
__global__ void BFS(CSRGraph graph, int  nowned, unsigned int * p_dist_current, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_BFS;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type wlvertex_end;
  const int _NP_CROSSOVER_WP = 32;
  const int _NP_CROSSOVER_TB = __kernel_tb_size;
  const int BLKSIZE = __kernel_tb_size;
  const int ITSIZE = BLKSIZE * 8;

  typedef cub::BlockScan<multiple_sum<2, index_type>, BLKSIZE> BlockScan;
  typedef union np_shared<BlockScan::TempStorage, index_type, struct tb_np, struct warp_np<__kernel_tb_size/32>, struct fg_np<ITSIZE> > npsTy;

  __shared__ npsTy nps ;
  wlvertex_end = roundup((*((volatile index_type *) (in_wl).dindex)), (blockDim.x));
  for (index_type wlvertex = 0 + tid; wlvertex < wlvertex_end; wlvertex += nthreads)
  {
    int src;
    bool pop;
    unsigned int sdist;
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    pop = (in_wl).pop_id(wlvertex, src);
    sdist = p_dist_current[src];
    struct NPInspector1 _np = {0,0,0,0,0,0};
    __shared__ struct { unsigned int sdist; } _np_closure [TB_SIZE];
    _np_closure[threadIdx.x].sdist = sdist;
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
      nps.tb.owner = MAX_TB_SIZE + 1;
    }
    __syncthreads();
    while (true)
    {
      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }
      __syncthreads();
      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {
        __syncthreads();
        break;
      }
      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }
      __syncthreads();
      int ns = nps.tb.start;
      int ne = nps.tb.size;
      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }
      assert(nps.tb.src < __kernel_tb_size);
      sdist = _np_closure[nps.tb.src].sdist;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type jj;
        jj = ns +_np_j;
        {
          index_type dst;
          unsigned int new_dist;
          unsigned int old_dist;
          dst = graph.getAbsDestination(jj);
          new_dist = 1 + sdist;
          old_dist = atomicMin(&p_dist_current[dst], new_dist);
          if (old_dist > new_dist)
          {
            index_type _start_30;
            _start_30 = (out_wl).setup_push_warp_one();;
            (out_wl).do_push(_start_30, 0, dst);
          }
        }
      }
      __syncthreads();
    }

    {
      const int warpid = threadIdx.x / 32;
      const int _np_laneid = cub::LaneId();
      while (__any(_np.size >= _NP_CROSSOVER_WP && _np.size < _NP_CROSSOVER_TB))
      {
        if (_np.size >= _NP_CROSSOVER_WP && _np.size < _NP_CROSSOVER_TB)
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
        sdist = _np_closure[nps.warp.src[warpid]].sdist;
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type jj;
          jj = _np_w_start +_np_ii;
          {
            index_type dst;
            unsigned int new_dist;
            unsigned int old_dist;
            dst = graph.getAbsDestination(jj);
            new_dist = 1 + sdist;
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
        index_type jj;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        sdist = _np_closure[nps.fg.src[_np_i]].sdist;
        jj= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          unsigned int new_dist;
          unsigned int old_dist;
          dst = graph.getAbsDestination(jj);
          new_dist = 1 + sdist;
          old_dist = atomicMin(&p_dist_current[dst], new_dist);
          if (old_dist > new_dist)
          {
            index_type _start_30;
            _start_30 = (out_wl).setup_push_warp_one();;
            (out_wl).do_push(_start_30, 0, dst);
          }
        }
      }
      _np.execute_round_done(ITSIZE);
      __syncthreads();
    }
    assert(threadIdx.x < __kernel_tb_size);
    sdist = _np_closure[threadIdx.x].sdist;
  }
}
void InitializeGraph_cuda(unsigned int local_src_node, unsigned int local_infinity, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  kernel_sizing(ctx->gg, blocks, threads);
  InitializeGraph <<<blocks, threads>>>(ctx->gg, ctx->nowned, local_infinity, local_src_node, ctx->dist_current.gpu_wr_ptr());
  check_cuda_kernel;
}
void BFS_cuda(struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  kernel_sizing(ctx->gg, blocks, threads);
  ctx->in_wl.update_gpu(ctx->shared_wl->num_in_items);
  ctx->out_wl.will_write();
  ctx->out_wl.reset();
  BFS <<<blocks, __tb_BFS>>>(ctx->gg, ctx->nowned, ctx->dist_current.gpu_wr_ptr(), ctx->in_wl, ctx->out_wl);
  check_cuda_kernel;
  ctx->out_wl.update_cpu();
  ctx->shared_wl->num_out_items = ctx->out_wl.nitems();
}