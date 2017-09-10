/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
unsigned int * P_NOUT;
float * P_RESIDUAL;
float * P_VALUE;
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
static const int __tb_PageRank = TB_SIZE;
static const int __tb_InitializeGraph = TB_SIZE;
__global__ void ResetGraph(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, unsigned int * p_nout, float * p_residual, float * p_value)
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
      p_residual[src] = 0;
    }
  }
  // FP: "9 -> 10;
}
__global__ void InitializeGraph(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, const float  local_alpha, unsigned int * p_nout, float * p_residual, float * p_value)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_InitializeGraph;
  float delta;
  index_type src_end;
  index_type src_rup;
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
  // FP: "6 -> 7;
  src_end = __end;
  src_rup = ((__begin) + roundup(((__end) - (__begin)), (blockDim.x)));
  for (index_type src = __begin + tid; src < src_rup; src += nthreads)
  {
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    // FP: "7 -> 8;
    bool pop  = src < __end;
    // FP: "8 -> 9;
    if (pop)
    {
      p_value[src] = local_alpha;
      p_nout[src] = graph.getOutDegree(src);
      if (p_nout[src] > 0)
      {
        delta = p_value[src]*(1-local_alpha)/p_nout[src];
      }
      else
      {
        pop = false;
      }
    }
    // FP: "15 -> 16;
    // FP: "18 -> 19;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "19 -> 20;
    __shared__ struct { float delta; } _np_closure [TB_SIZE];
    // FP: "20 -> 21;
    _np_closure[threadIdx.x].delta = delta;
    // FP: "21 -> 22;
    if (pop)
    {
      _np.size = (graph).getOutDegree(src);
      _np.start = (graph).getFirstEdge(src);
    }
    // FP: "24 -> 25;
    // FP: "25 -> 26;
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    // FP: "26 -> 27;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    // FP: "27 -> 28;
    if (threadIdx.x == 0)
    {
      nps.tb.owner = MAX_TB_SIZE + 1;
    }
    // FP: "30 -> 31;
    __syncthreads();
    // FP: "31 -> 32;
    while (true)
    {
      // FP: "32 -> 33;
      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }
      // FP: "35 -> 36;
      __syncthreads();
      // FP: "36 -> 37;
      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {
        // FP: "37 -> 38;
        __syncthreads();
        // FP: "38 -> 39;
        break;
      }
      // FP: "40 -> 41;
      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }
      // FP: "43 -> 44;
      __syncthreads();
      // FP: "44 -> 45;
      int ns = nps.tb.start;
      int ne = nps.tb.size;
      // FP: "45 -> 46;
      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }
      // FP: "48 -> 49;
      assert(nps.tb.src < __kernel_tb_size);
      delta = _np_closure[nps.tb.src].delta;
      // FP: "49 -> 50;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type nbr;
        nbr = ns +_np_j;
        {
          index_type dst;
          dst = graph.getAbsDestination(nbr);
          atomicAdd(&p_residual[dst], delta);
        }
      }
      // FP: "56 -> 57;
      __syncthreads();
    }
    // FP: "58 -> 59;

    // FP: "59 -> 60;
    {
      const int warpid = threadIdx.x / 32;
      // FP: "60 -> 61;
      const int _np_laneid = cub::LaneId();
      // FP: "61 -> 62;
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
        delta = _np_closure[nps.warp.src[warpid]].delta;
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type nbr;
          nbr = _np_w_start +_np_ii;
          {
            index_type dst;
            dst = graph.getAbsDestination(nbr);
            atomicAdd(&p_residual[dst], delta);
          }
        }
      }
      // FP: "78 -> 79;
      __syncthreads();
      // FP: "79 -> 80;
    }

    // FP: "80 -> 81;
    __syncthreads();
    // FP: "81 -> 82;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "82 -> 83;
    while (_np.work())
    {
      // FP: "83 -> 84;
      int _np_i =0;
      // FP: "84 -> 85;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "85 -> 86;
      __syncthreads();
      // FP: "86 -> 87;

      // FP: "87 -> 88;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type nbr;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        delta = _np_closure[nps.fg.src[_np_i]].delta;
        nbr= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          dst = graph.getAbsDestination(nbr);
          atomicAdd(&p_residual[dst], delta);
        }
      }
      // FP: "95 -> 96;
      _np.execute_round_done(ITSIZE);
      // FP: "96 -> 97;
      __syncthreads();
    }
    // FP: "98 -> 99;
    assert(threadIdx.x < __kernel_tb_size);
    delta = _np_closure[threadIdx.x].delta;
  }
  // FP: "101 -> 102;
}
__global__ void PageRank(CSRGraph graph, unsigned int __nowned, const float  local_alpha, float local_tolerance, unsigned int * p_nout, float * p_residual, float * p_value, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_PageRank;
  if (tid == 0)
    in_wl.reset_next_slot();

  float residual_old;
  float delta;
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
  // FP: "6 -> 7;
  // FP: "7 -> 8;
  wlvertex_end = *((volatile index_type *) (in_wl).dindex);
  wlvertex_rup = ((0) + roundup(((*((volatile index_type *) (in_wl).dindex)) - (0)), (blockDim.x)));
  for (index_type wlvertex = 0 + tid; wlvertex < wlvertex_rup; wlvertex += nthreads)
  {
    int src;
    bool pop;
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    // FP: "8 -> 9;
    // FP: "9 -> 10;
    // FP: "10 -> 11;
    pop = (in_wl).pop_id(wlvertex, src);
    // FP: "11 -> 12;
    if (pop)
    {
      residual_old = atomicExch(&p_residual[src], 0.0);
      p_value[src] += residual_old;
      if (p_nout[src] > 0)
      {
        delta = residual_old*(1-local_alpha)/p_nout[src];
      }
      else
      {
        pop = false;
      }
    }
    // FP: "18 -> 19;
    // FP: "21 -> 22;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "22 -> 23;
    __shared__ struct { float delta; } _np_closure [TB_SIZE];
    // FP: "23 -> 24;
    _np_closure[threadIdx.x].delta = delta;
    // FP: "24 -> 25;
    if (pop)
    {
      _np.size = (graph).getOutDegree(src);
      _np.start = (graph).getFirstEdge(src);
    }
    // FP: "27 -> 28;
    // FP: "28 -> 29;
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    // FP: "29 -> 30;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    // FP: "30 -> 31;
    if (threadIdx.x == 0)
    {
      nps.tb.owner = MAX_TB_SIZE + 1;
    }
    // FP: "33 -> 34;
    __syncthreads();
    // FP: "34 -> 35;
    while (true)
    {
      // FP: "35 -> 36;
      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }
      // FP: "38 -> 39;
      __syncthreads();
      // FP: "39 -> 40;
      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {
        // FP: "40 -> 41;
        __syncthreads();
        // FP: "41 -> 42;
        break;
      }
      // FP: "43 -> 44;
      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }
      // FP: "46 -> 47;
      __syncthreads();
      // FP: "47 -> 48;
      int ns = nps.tb.start;
      int ne = nps.tb.size;
      // FP: "48 -> 49;
      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }
      // FP: "51 -> 52;
      assert(nps.tb.src < __kernel_tb_size);
      delta = _np_closure[nps.tb.src].delta;
      // FP: "52 -> 53;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type nbr;
        nbr = ns +_np_j;
        {
          index_type dst;
          float dst_residual_old;
          dst = graph.getAbsDestination(nbr);
          dst_residual_old = atomicAdd(&p_residual[dst], delta);
          if ((dst_residual_old <= local_tolerance) && ((dst_residual_old + delta) >= local_tolerance))
          {
            index_type _start_75;
            _start_75 = (out_wl).setup_push_warp_one();;
            (out_wl).do_push(_start_75, 0, dst);
          }
        }
      }
      // FP: "65 -> 66;
      __syncthreads();
    }
    // FP: "67 -> 68;

    // FP: "68 -> 69;
    {
      const int warpid = threadIdx.x / 32;
      // FP: "69 -> 70;
      const int _np_laneid = cub::LaneId();
      // FP: "70 -> 71;
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
              index_type _start_75;
              _start_75 = (out_wl).setup_push_warp_one();;
              (out_wl).do_push(_start_75, 0, dst);
            }
          }
        }
      }
      // FP: "93 -> 94;
      __syncthreads();
      // FP: "94 -> 95;
    }

    // FP: "95 -> 96;
    __syncthreads();
    // FP: "96 -> 97;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "97 -> 98;
    while (_np.work())
    {
      // FP: "98 -> 99;
      int _np_i =0;
      // FP: "99 -> 100;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "100 -> 101;
      __syncthreads();
      // FP: "101 -> 102;

      // FP: "102 -> 103;
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
            index_type _start_75;
            _start_75 = (out_wl).setup_push_warp_one();;
            (out_wl).do_push(_start_75, 0, dst);
          }
        }
      }
      // FP: "116 -> 117;
      _np.execute_round_done(ITSIZE);
      // FP: "117 -> 118;
      __syncthreads();
    }
    // FP: "119 -> 120;
    assert(threadIdx.x < __kernel_tb_size);
    delta = _np_closure[threadIdx.x].delta;
  }
  // FP: "122 -> 123;
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
  ResetGraph <<<blocks, threads>>>(ctx->gg, ctx->nowned, __begin, __end, ctx->nout.data.gpu_wr_ptr(), ctx->residual.data.gpu_wr_ptr(), ctx->value.data.gpu_wr_ptr());
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
  InitializeGraph <<<blocks, __tb_InitializeGraph>>>(ctx->gg, ctx->nowned, __begin, __end, local_alpha, ctx->nout.data.gpu_wr_ptr(), ctx->residual.data.gpu_wr_ptr(), ctx->value.data.gpu_wr_ptr());
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
void PageRank_cuda(const float & local_alpha, float local_tolerance, struct CUDA_Context * ctx)
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
  PageRank <<<blocks, __tb_PageRank>>>(ctx->gg, ctx->nowned, local_alpha, local_tolerance, ctx->nout.data.gpu_wr_ptr(), ctx->residual.data.gpu_wr_ptr(), ctx->value.data.gpu_wr_ptr(), ctx->in_wl, ctx->out_wl);
  // FP: "8 -> 9;
  check_cuda_kernel;
  // FP: "9 -> 10;
  ctx->out_wl.update_cpu();
  // FP: "10 -> 11;
  ctx->shared_wl->num_out_items = ctx->out_wl.nitems();
  // FP: "11 -> 12;
}