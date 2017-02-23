/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
unsigned int * P_COMP_CURRENT;
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
static const int __tb_ConnectedComp = TB_SIZE;
__global__ void InitializeGraph(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, unsigned int * p_comp_current)
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
      p_comp_current[src] = graph.node_data[src];
    }
  }
  // FP: "7 -> 8;
}
__global__ void ConnectedComp(CSRGraph graph, unsigned int __nowned, unsigned int * p_comp_current, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_ConnectedComp;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type wlvertex_end;
  index_type wlvertex_rup;
  // FP: "1 -> 2;
  const int _NP_CROSSOVER_WP = 32;
  const int _NP_CROSSOVER_TB = __kernel_tb_size;
  // FP: "2 -> 3;
  const int BLKSIZE = __kernel_tb_size;
  const int ITSIZE = BLKSIZE * 8;
  // FP: "3 -> 4;

  typedef cub::BlockScan<multiple_sum<2, index_type>, BLKSIZE> BlockScan;
  typedef union np_shared<BlockScan::TempStorage, index_type, struct tb_np, struct warp_np<__kernel_tb_size/32>, struct fg_np<ITSIZE> > npsTy;

  // FP: "4 -> 5;
  __shared__ npsTy nps ;
  // FP: "5 -> 6;
  wlvertex_end = *((volatile index_type *) (in_wl).dindex);
  wlvertex_rup = ((0) + roundup(((*((volatile index_type *) (in_wl).dindex)) - (0)), (blockDim.x)));
  for (index_type wlvertex = 0 + tid; wlvertex < wlvertex_rup; wlvertex += nthreads)
  {
    int src;
    bool pop;
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    // FP: "6 -> 7;
    // FP: "7 -> 8;
    // FP: "8 -> 9;
    pop = (in_wl).pop_id(wlvertex, src);
    // FP: "9 -> 10;
    if (pop)
    {
    }
    // FP: "11 -> 12;
    // FP: "14 -> 15;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "15 -> 16;
    __shared__ struct { int src; } _np_closure [TB_SIZE];
    // FP: "16 -> 17;
    _np_closure[threadIdx.x].src = src;
    // FP: "17 -> 18;
    if (pop)
    {
      _np.size = (graph).getOutDegree(src);
      _np.start = (graph).getFirstEdge(src);
    }
    // FP: "20 -> 21;
    // FP: "21 -> 22;
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    // FP: "22 -> 23;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    // FP: "23 -> 24;
    if (threadIdx.x == 0)
    {
      nps.tb.owner = MAX_TB_SIZE + 1;
    }
    // FP: "26 -> 27;
    __syncthreads();
    // FP: "27 -> 28;
    while (true)
    {
      // FP: "28 -> 29;
      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }
      // FP: "31 -> 32;
      __syncthreads();
      // FP: "32 -> 33;
      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {
        // FP: "33 -> 34;
        __syncthreads();
        // FP: "34 -> 35;
        break;
      }
      // FP: "36 -> 37;
      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }
      // FP: "39 -> 40;
      __syncthreads();
      // FP: "40 -> 41;
      int ns = nps.tb.start;
      int ne = nps.tb.size;
      // FP: "41 -> 42;
      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }
      // FP: "44 -> 45;
      assert(nps.tb.src < __kernel_tb_size);
      src = _np_closure[nps.tb.src].src;
      // FP: "45 -> 46;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type jj;
        jj = ns +_np_j;
        {
          index_type dst;
          unsigned int new_dist;
          unsigned int old_dist;
          dst = graph.getAbsDestination(jj);
          new_dist = p_comp_current[src];
          old_dist = atomicMin(&p_comp_current[dst], new_dist);
          if (old_dist > new_dist)
          {
            index_type _start_39;
            _start_39 = (out_wl).setup_push_warp_one();;
            (out_wl).do_push(_start_39, 0, dst);
          }
        }
      }
      // FP: "60 -> 61;
      __syncthreads();
    }
    // FP: "62 -> 63;

    // FP: "63 -> 64;
    {
      const int warpid = threadIdx.x / 32;
      // FP: "64 -> 65;
      const int _np_laneid = cub::LaneId();
      // FP: "65 -> 66;
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
        src = _np_closure[nps.warp.src[warpid]].src;
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type jj;
          jj = _np_w_start +_np_ii;
          {
            index_type dst;
            unsigned int new_dist;
            unsigned int old_dist;
            dst = graph.getAbsDestination(jj);
            new_dist = p_comp_current[src];
            old_dist = atomicMin(&p_comp_current[dst], new_dist);
            if (old_dist > new_dist)
            {
              index_type _start_39;
              _start_39 = (out_wl).setup_push_warp_one();;
              (out_wl).do_push(_start_39, 0, dst);
            }
          }
        }
      }
      // FP: "90 -> 91;
      __syncthreads();
      // FP: "91 -> 92;
    }

    // FP: "92 -> 93;
    __syncthreads();
    // FP: "93 -> 94;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "94 -> 95;
    while (_np.work())
    {
      // FP: "95 -> 96;
      int _np_i =0;
      // FP: "96 -> 97;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "97 -> 98;
      __syncthreads();
      // FP: "98 -> 99;

      // FP: "99 -> 100;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type jj;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        jj= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          unsigned int new_dist;
          unsigned int old_dist;
          dst = graph.getAbsDestination(jj);
          new_dist = p_comp_current[src];
          old_dist = atomicMin(&p_comp_current[dst], new_dist);
          if (old_dist > new_dist)
          {
            index_type _start_39;
            _start_39 = (out_wl).setup_push_warp_one();;
            (out_wl).do_push(_start_39, 0, dst);
          }
        }
      }
      // FP: "115 -> 116;
      _np.execute_round_done(ITSIZE);
      // FP: "116 -> 117;
      __syncthreads();
    }
    // FP: "118 -> 119;
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
  }
  // FP: "120 -> 121;
}
void InitializeGraph_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  InitializeGraph <<<blocks, threads>>>(ctx->gg, ctx->nowned, __begin, __end, ctx->comp_current.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void InitializeGraph_all_cuda(struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  InitializeGraph_cuda(0, ctx->nowned, ctx);
  // FP: "2 -> 3;
}
void ConnectedComp_cuda(struct CUDA_Context * ctx)
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
  ConnectedComp <<<blocks, __tb_ConnectedComp>>>(ctx->gg, ctx->nowned, ctx->comp_current.data.gpu_wr_ptr(), ctx->in_wl, ctx->out_wl);
  // FP: "8 -> 9;
  check_cuda_kernel;
  // FP: "9 -> 10;
  ctx->out_wl.update_cpu();
  // FP: "10 -> 11;
  ctx->shared_wl->num_out_items = ctx->out_wl.nitems();
  // FP: "11 -> 12;
}