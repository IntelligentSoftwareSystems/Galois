/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
int * P_NOUT;
float * P_SUM;
float * P_VALUE;
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
static const int __tb_PageRank_partial = TB_SIZE;
static const int __tb_InitializeGraph = TB_SIZE;
__global__ void ResetGraph(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, int * p_nout, float * p_value)
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
    }
  }
  // FP: "8 -> 9;
}
__global__ void InitializeGraph(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, const float  local_alpha, int * p_nout, float * p_value)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_InitializeGraph;
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
      p_value[src] = local_alpha;
    }
    // FP: "10 -> 11;
    // FP: "13 -> 14;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "14 -> 15;
    __shared__ struct { ; } _np_closure [TB_SIZE];
    // FP: "15 -> 16;
    // FP: "16 -> 17;
    if (pop)
    {
      _np.size = (graph).getOutDegree(src);
      _np.start = (graph).getFirstEdge(src);
    }
    // FP: "19 -> 20;
    // FP: "20 -> 21;
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    // FP: "21 -> 22;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    // FP: "22 -> 23;
    if (threadIdx.x == 0)
    {
      nps.tb.owner = MAX_TB_SIZE + 1;
    }
    // FP: "25 -> 26;
    __syncthreads();
    // FP: "26 -> 27;
    while (true)
    {
      // FP: "27 -> 28;
      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }
      // FP: "30 -> 31;
      __syncthreads();
      // FP: "31 -> 32;
      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {
        // FP: "32 -> 33;
        __syncthreads();
        // FP: "33 -> 34;
        break;
      }
      // FP: "35 -> 36;
      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }
      // FP: "38 -> 39;
      __syncthreads();
      // FP: "39 -> 40;
      int ns = nps.tb.start;
      int ne = nps.tb.size;
      // FP: "40 -> 41;
      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }
      // FP: "43 -> 44;
      assert(nps.tb.src < __kernel_tb_size);
      // FP: "44 -> 45;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type nbr;
        nbr = ns +_np_j;
        {
          index_type dst;
          dst = graph.getAbsDestination(nbr);
          atomicAdd(&p_nout[dst], 1);
        }
      }
      // FP: "51 -> 52;
      __syncthreads();
    }
    // FP: "53 -> 54;

    // FP: "54 -> 55;
    {
      const int warpid = threadIdx.x / 32;
      // FP: "55 -> 56;
      const int _np_laneid = cub::LaneId();
      // FP: "56 -> 57;
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
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type nbr;
          nbr = _np_w_start +_np_ii;
          {
            index_type dst;
            dst = graph.getAbsDestination(nbr);
            atomicAdd(&p_nout[dst], 1);
          }
        }
      }
      // FP: "73 -> 74;
      __syncthreads();
      // FP: "74 -> 75;
    }

    // FP: "75 -> 76;
    __syncthreads();
    // FP: "76 -> 77;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "77 -> 78;
    while (_np.work())
    {
      // FP: "78 -> 79;
      int _np_i =0;
      // FP: "79 -> 80;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "80 -> 81;
      __syncthreads();
      // FP: "81 -> 82;

      // FP: "82 -> 83;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type nbr;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        nbr= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          dst = graph.getAbsDestination(nbr);
          atomicAdd(&p_nout[dst], 1);
        }
      }
      // FP: "90 -> 91;
      _np.execute_round_done(ITSIZE);
      // FP: "91 -> 92;
      __syncthreads();
    }
    // FP: "93 -> 94;
    assert(threadIdx.x < __kernel_tb_size);
  }
  // FP: "95 -> 96;
}
__global__ void PageRank_partial(CSRGraph graph, DynamicBitset *is_updated, unsigned int __nowned, unsigned int __begin, unsigned int __end, int * p_nout, float * p_sum, float * p_value)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_PageRank_partial;
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
      p_sum[src] = 0;
      is_updated->set(src);
    }
    // FP: "10 -> 11;
    // FP: "13 -> 14;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "14 -> 15;
    __shared__ struct { index_type src; } _np_closure [TB_SIZE];
    // FP: "15 -> 16;
    _np_closure[threadIdx.x].src = src;
    // FP: "16 -> 17;
    if (pop)
    {
      _np.size = (graph).getOutDegree(src);
      _np.start = (graph).getFirstEdge(src);
    }
    // FP: "19 -> 20;
    // FP: "20 -> 21;
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    // FP: "21 -> 22;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    // FP: "22 -> 23;
    if (threadIdx.x == 0)
    {
      nps.tb.owner = MAX_TB_SIZE + 1;
    }
    // FP: "25 -> 26;
    __syncthreads();
    // FP: "26 -> 27;
    while (true)
    {
      // FP: "27 -> 28;
      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }
      // FP: "30 -> 31;
      __syncthreads();
      // FP: "31 -> 32;
      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {
        // FP: "32 -> 33;
        __syncthreads();
        // FP: "33 -> 34;
        break;
      }
      // FP: "35 -> 36;
      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }
      // FP: "38 -> 39;
      __syncthreads();
      // FP: "39 -> 40;
      int ns = nps.tb.start;
      int ne = nps.tb.size;
      // FP: "40 -> 41;
      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }
      // FP: "43 -> 44;
      assert(nps.tb.src < __kernel_tb_size);
      src = _np_closure[nps.tb.src].src;
      // FP: "44 -> 45;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type nbr;
        nbr = ns +_np_j;
        {
          index_type dst;
          unsigned int dnout;
          dst = graph.getAbsDestination(nbr);
          dnout = p_nout[dst];
          if (dnout > 0)
          {
            atomicAdd(&p_sum[src], p_value[dst]/dnout);
          }
        }
      }
      // FP: "55 -> 56;
      __syncthreads();
    }
    // FP: "57 -> 58;

    // FP: "58 -> 59;
    {
      const int warpid = threadIdx.x / 32;
      // FP: "59 -> 60;
      const int _np_laneid = cub::LaneId();
      // FP: "60 -> 61;
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
          index_type nbr;
          nbr = _np_w_start +_np_ii;
          {
            index_type dst;
            unsigned int dnout;
            dst = graph.getAbsDestination(nbr);
            dnout = p_nout[dst];
            if (dnout > 0)
            {
              atomicAdd(&p_sum[src], p_value[dst]/dnout);
            }
          }
        }
      }
      // FP: "81 -> 82;
      __syncthreads();
      // FP: "82 -> 83;
    }

    // FP: "83 -> 84;
    __syncthreads();
    // FP: "84 -> 85;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "85 -> 86;
    while (_np.work())
    {
      // FP: "86 -> 87;
      int _np_i =0;
      // FP: "87 -> 88;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "88 -> 89;
      __syncthreads();
      // FP: "89 -> 90;

      // FP: "90 -> 91;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type nbr;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        nbr= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          unsigned int dnout;
          dst = graph.getAbsDestination(nbr);
          dnout = p_nout[dst];
          if (dnout > 0)
          {
            atomicAdd(&p_sum[src], p_value[dst]/dnout);
          }
        }
      }
      // FP: "102 -> 103;
      _np.execute_round_done(ITSIZE);
      // FP: "103 -> 104;
      __syncthreads();
    }
    // FP: "105 -> 106;
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
  }
  // FP: "107 -> 108;
}
__global__ void PageRank(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, const float  local_alpha, float local_tolerance, float * p_sum, float * p_value, Sum ret_val)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  typedef cub::BlockReduce<int, TB_SIZE> _br;
  __shared__ _br::TempStorage _ts;
  ret_val.thread_entry();
  float pr_value;
  float diff;
  index_type src_end;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      pr_value = p_sum[src]*(1.0 - local_alpha) + local_alpha;
      diff = pr_value - p_value[src];
      if (diff > local_tolerance)
      {
        p_value[src] = pr_value;
        ret_val.do_return( 1);
        continue;
      }
    }
  }
  ret_val.thread_exit<_br>(_ts);
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
  ResetGraph <<<blocks, threads>>>(ctx->gg, ctx->nowned, __begin, __end, ctx->nout.data.gpu_wr_ptr(), ctx->value.data.gpu_wr_ptr());
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
  InitializeGraph <<<blocks, __tb_InitializeGraph>>>(ctx->gg, ctx->nowned, __begin, __end, local_alpha, ctx->nout.data.gpu_wr_ptr(), ctx->value.data.gpu_wr_ptr());
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
void PageRank_partial_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  PageRank_partial <<<blocks, __tb_PageRank_partial>>>(ctx->gg, ctx->sum.is_updated.gpu_rd_ptr(), ctx->nowned, __begin, __end, ctx->nout.data.gpu_wr_ptr(), ctx->sum.data.gpu_wr_ptr(), ctx->value.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void PageRank_partial_all_cuda(struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  PageRank_partial_cuda(0, ctx->nowned, ctx);
  // FP: "2 -> 3;
}
void PageRank_cuda(unsigned int  __begin, unsigned int  __end, int & __retval, const float & local_alpha, float local_tolerance, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  Shared<int> retval = Shared<int>(1);
  Sum _rv;
  *(retval.cpu_wr_ptr()) = 0;
  _rv.rv = retval.gpu_wr_ptr();
  PageRank <<<blocks, threads>>>(ctx->gg, ctx->nowned, __begin, __end, local_alpha, local_tolerance, ctx->sum.data.gpu_wr_ptr(), ctx->value.data.gpu_wr_ptr(), _rv);
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
  __retval = *(retval.cpu_rd_ptr());
  // FP: "7 -> 8;
}
void PageRank_all_cuda(int & __retval, const float & local_alpha, float local_tolerance, struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  PageRank_cuda(0, ctx->nowned, __retval, local_alpha, local_tolerance, ctx);
  // FP: "2 -> 3;
}
