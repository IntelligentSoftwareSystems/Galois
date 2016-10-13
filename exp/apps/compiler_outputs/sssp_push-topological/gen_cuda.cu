/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
unsigned int * P_DIST_CURRENT;
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
static const int __tb_SSSP = TB_SIZE;
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
__global__ void SSSP(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, unsigned int * p_dist_current, Sum sum_retval)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_SSSP;
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
  src_end = __end;
  src_rup = ((__begin) + roundup(((__end) - (__begin)), (blockDim.x)));
  for (index_type src = __begin + tid; src < src_rup; src += nthreads)
  {
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    // FP: "6 -> 7;
    bool pop  = src < __end;
    // FP: "7 -> 8;
    if (pop)
    {
    }
    // FP: "9 -> 10;
    // FP: "12 -> 13;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "13 -> 14;
    __shared__ struct { index_type src; } _np_closure [TB_SIZE];
    // FP: "14 -> 15;
    _np_closure[threadIdx.x].src = src;
    // FP: "15 -> 16;
    if (pop)
    {
      _np.size = (graph).getOutDegree(src);
      _np.start = (graph).getFirstEdge(src);
    }
    // FP: "18 -> 19;
    // FP: "19 -> 20;
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    // FP: "20 -> 21;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    // FP: "21 -> 22;
    if (threadIdx.x == 0)
    {
      nps.tb.owner = MAX_TB_SIZE + 1;
    }
    // FP: "24 -> 25;
    __syncthreads();
    // FP: "25 -> 26;
    while (true)
    {
      // FP: "26 -> 27;
      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }
      // FP: "29 -> 30;
      __syncthreads();
      // FP: "30 -> 31;
      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {
        // FP: "31 -> 32;
        __syncthreads();
        // FP: "32 -> 33;
        break;
      }
      // FP: "34 -> 35;
      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }
      // FP: "37 -> 38;
      __syncthreads();
      // FP: "38 -> 39;
      int ns = nps.tb.start;
      int ne = nps.tb.size;
      // FP: "39 -> 40;
      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }
      // FP: "42 -> 43;
      assert(nps.tb.src < __kernel_tb_size);
      src = _np_closure[nps.tb.src].src;
      // FP: "43 -> 44;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type jj;
        jj = ns +_np_j;
        {
          index_type dst;
          unsigned int new_dist;
          unsigned int old_dist;
          dst = graph.getAbsDestination(jj);
          new_dist = graph.getAbsWeight(jj) + p_dist_current[src];
          old_dist = atomicMin(&p_dist_current[dst], new_dist);
          if (old_dist > new_dist)
          {
            sum_retval.do_return( 1);
          }
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
            new_dist = graph.getAbsWeight(jj) + p_dist_current[src];
            old_dist = atomicMin(&p_dist_current[dst], new_dist);
            if (old_dist > new_dist)
            {
              sum_retval.do_return( 1);
            }
          }
        }
      }
      // FP: "84 -> 85;
      __syncthreads();
      // FP: "85 -> 86;
    }

    // FP: "86 -> 87;
    __syncthreads();
    // FP: "87 -> 88;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "88 -> 89;
    while (_np.work())
    {
      // FP: "89 -> 90;
      int _np_i =0;
      // FP: "90 -> 91;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "91 -> 92;
      __syncthreads();
      // FP: "92 -> 93;

      // FP: "93 -> 94;
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
          new_dist = graph.getAbsWeight(jj) + p_dist_current[src];
          old_dist = atomicMin(&p_dist_current[dst], new_dist);
          if (old_dist > new_dist)
          {
            sum_retval.do_return( 1);
          }
        }
      }
      // FP: "107 -> 108;
      _np.execute_round_done(ITSIZE);
      // FP: "108 -> 109;
      __syncthreads();
    }
    // FP: "110 -> 111;
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
  }
  // FP: "112 -> 113;
}
void InitializeGraph_cuda(unsigned int  __begin, unsigned int  __end, const unsigned int & local_infinity, unsigned int local_src_node, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(ctx->gg, blocks, threads);
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
void SSSP_cuda(unsigned int  __begin, unsigned int  __end, int & __retval, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(ctx->gg, blocks, threads);
  // FP: "4 -> 5;
  *(ctx->p_retval.cpu_wr_ptr()) = __retval;
  // FP: "5 -> 6;
  ctx->sum_retval.rv = ctx->p_retval.gpu_wr_ptr();
  // FP: "6 -> 7;
  SSSP <<<blocks, __tb_SSSP>>>(ctx->gg, ctx->nowned, __begin, __end, ctx->dist_current.gpu_wr_ptr(), ctx->sum_retval);
  // FP: "7 -> 8;
  check_cuda_kernel;
  // FP: "8 -> 9;
  __retval = *(ctx->p_retval.cpu_rd_ptr());
  // FP: "9 -> 10;
}
void SSSP_all_cuda(int & __retval, struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  SSSP_cuda(0, ctx->nowned, __retval, ctx);
  // FP: "2 -> 3;
}