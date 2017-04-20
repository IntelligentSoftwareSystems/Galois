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
static const int __tb_FirstItr_PageRank = TB_SIZE;
static const int __tb_PageRank = TB_SIZE;
static const int __tb_InitializeGraph = TB_SIZE;
static const int __tb_InitializeGraphNout = TB_SIZE;
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
__global__ void InitializeGraphNout(CSRGraph graph, DynamicBitset *is_updated, unsigned int __nowned, unsigned int __begin, unsigned int __end, unsigned int * p_nout)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_InitializeGraphNout;
  index_type src_end;
  // FP: "1 -> 2;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      atomicAdd(&p_nout[src], graph.getOutDegree(src));
      is_updated->set(src);
    }
  }
  // FP: "8 -> 9;
}
__global__ void InitializeGraph(CSRGraph graph, DynamicBitset *is_updated, unsigned int __nowned, unsigned int __begin, unsigned int __end, const float  local_alpha, unsigned int * p_nout, float * p_residual, float * p_value)
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
          is_updated->set(dst);
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
            is_updated->set(dst);
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
          is_updated->set(dst);
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
__global__ void PageRankCopy(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, float * p_residual, float * p_residual_old)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_InitializeGraphNout;
  index_type src_end;
  // FP: "1 -> 2;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      p_residual_old[src] += p_residual[src];
      p_residual[src] = 0;
    }
  }
  // FP: "8 -> 9;
}
__global__ void FirstItr_PageRank(CSRGraph graph, DynamicBitset *is_updated, unsigned int __nowned, unsigned int __begin, unsigned int __end, const float  local_alpha, unsigned int * p_nout, float * p_residual, float * p_value)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_FirstItr_PageRank;
  float residual_old;
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
  // FP: "7 -> 8;
  src_end = __end;
  src_rup = ((__begin) + roundup(((__end) - (__begin)), (blockDim.x)));
  for (index_type src = __begin + tid; src < src_rup; src += nthreads)
  {
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    // FP: "8 -> 9;
    bool pop  = src < __end;
    // FP: "9 -> 10;
    if (pop)
    {
      if (p_residual[src] > 0)
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
      else
      {
        pop = false;
      }
    }
    // FP: "16 -> 17;
    // FP: "19 -> 20;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "20 -> 21;
    __shared__ struct { float delta; } _np_closure [TB_SIZE];
    // FP: "21 -> 22;
    _np_closure[threadIdx.x].delta = delta;
    // FP: "22 -> 23;
    if (pop)
    {
      _np.size = (graph).getOutDegree(src);
      _np.start = (graph).getFirstEdge(src);
    }
    // FP: "25 -> 26;
    // FP: "26 -> 27;
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    // FP: "27 -> 28;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    // FP: "28 -> 29;
    if (threadIdx.x == 0)
    {
      nps.tb.owner = MAX_TB_SIZE + 1;
    }
    // FP: "31 -> 32;
    __syncthreads();
    // FP: "32 -> 33;
    while (true)
    {
      // FP: "33 -> 34;
      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }
      // FP: "36 -> 37;
      __syncthreads();
      // FP: "37 -> 38;
      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {
        // FP: "38 -> 39;
        __syncthreads();
        // FP: "39 -> 40;
        break;
      }
      // FP: "41 -> 42;
      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }
      // FP: "44 -> 45;
      __syncthreads();
      // FP: "45 -> 46;
      int ns = nps.tb.start;
      int ne = nps.tb.size;
      // FP: "46 -> 47;
      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }
      // FP: "49 -> 50;
      assert(nps.tb.src < __kernel_tb_size);
      delta = _np_closure[nps.tb.src].delta;
      // FP: "50 -> 51;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type nbr;
        nbr = ns +_np_j;
        {
          index_type dst;
          float dst_residual_old;
          dst = graph.getAbsDestination(nbr);
          dst_residual_old = atomicAdd(&p_residual[dst], delta);
          is_updated->set(dst);
        }
      }
      // FP: "58 -> 59;
      __syncthreads();
    }
    // FP: "60 -> 61;

    // FP: "61 -> 62;
    {
      const int warpid = threadIdx.x / 32;
      // FP: "62 -> 63;
      const int _np_laneid = cub::LaneId();
      // FP: "63 -> 64;
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
            is_updated->set(dst);
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
        delta = _np_closure[nps.fg.src[_np_i]].delta;
        nbr= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          float dst_residual_old;
          dst = graph.getAbsDestination(nbr);
          dst_residual_old = atomicAdd(&p_residual[dst], delta);
          is_updated->set(dst);
        }
      }
      // FP: "99 -> 100;
      _np.execute_round_done(ITSIZE);
      // FP: "100 -> 101;
      __syncthreads();
    }
    // FP: "102 -> 103;
    assert(threadIdx.x < __kernel_tb_size);
    delta = _np_closure[threadIdx.x].delta;
  }
  // FP: "105 -> 106;
}
__global__ void PageRank(CSRGraph graph, DynamicBitset *is_updated, unsigned int __nowned, unsigned int __begin, unsigned int __end, const float  local_alpha, float local_tolerance, unsigned int * p_nout, float * p_residual, float * p_value, Sum ret_val)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_PageRank;
  typedef cub::BlockReduce<int, TB_SIZE> _br;
  __shared__ _br::TempStorage _ts;
  ret_val.thread_entry();
  float residual_old;
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
  // FP: "7 -> 8;
  src_end = __end;
  src_rup = ((__begin) + roundup(((__end) - (__begin)), (blockDim.x)));
  for (index_type src = __begin + tid; src < src_rup; src += nthreads)
  {
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    // FP: "8 -> 9;
    bool pop  = src < __end;
    // FP: "9 -> 10;
    if (pop)
    {
      if (p_residual[src] > local_tolerance)
      {
        residual_old = atomicExch(&p_residual[src], 0.0);
        p_value[src] += residual_old;
        if (p_nout[src] > 0)
        {
          delta = residual_old*(1-local_alpha)/p_nout[src];
          ret_val.do_return( 1);
        }
        else
        {
          pop = false;
        }
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
          is_updated->set(dst);
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
            is_updated->set(dst);
          }
        }
      }
      // FP: "83 -> 84;
      __syncthreads();
      // FP: "84 -> 85;
    }

    // FP: "85 -> 86;
    __syncthreads();
    // FP: "86 -> 87;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "87 -> 88;
    while (_np.work())
    {
      // FP: "88 -> 89;
      int _np_i =0;
      // FP: "89 -> 90;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "90 -> 91;
      __syncthreads();
      // FP: "91 -> 92;

      // FP: "92 -> 93;
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
          is_updated->set(dst);
        }
      }
      // FP: "101 -> 102;
      _np.execute_round_done(ITSIZE);
      // FP: "102 -> 103;
      __syncthreads();
    }
    // FP: "104 -> 105;
    assert(threadIdx.x < __kernel_tb_size);
    delta = _np_closure[threadIdx.x].delta;
    // FP: "105 -> 106;
    // FP: "106 -> 107;
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
void InitializeGraphNout_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  InitializeGraphNout <<<blocks, __tb_InitializeGraphNout>>>(ctx->gg, ctx->nout.is_updated.gpu_rd_ptr(), ctx->nowned, __begin, __end, ctx->nout.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void InitializeGraphNout_all_cuda(struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  InitializeGraphNout_cuda(0, ctx->nowned, ctx);
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
  InitializeGraph <<<blocks, __tb_InitializeGraph>>>(ctx->gg, ctx->residual.is_updated.gpu_rd_ptr(), ctx->nowned, __begin, __end, local_alpha, ctx->nout.data.gpu_wr_ptr(), ctx->residual.data.gpu_wr_ptr(), ctx->value.data.gpu_wr_ptr());
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
void PageRankCopy_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  PageRankCopy <<<blocks, __tb_PageRank>>>(ctx->gg, ctx->nowned, __begin, __end, ctx->residual.data.gpu_wr_ptr(), ctx->residual_old.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void PageRankCopy_all_cuda(struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  PageRankCopy_cuda(0, ctx->nowned, ctx);
  // FP: "2 -> 3;
}
void FirstItr_PageRank_cuda(unsigned int  __begin, unsigned int  __end, const float & local_alpha, float local_tolerance, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  FirstItr_PageRank <<<blocks, __tb_FirstItr_PageRank>>>(ctx->gg, ctx->residual.is_updated.gpu_rd_ptr(), ctx->nowned, __begin, __end, local_alpha, ctx->nout.data.gpu_wr_ptr(), ctx->residual.data.gpu_wr_ptr(), ctx->value.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void FirstItr_PageRank_all_cuda(const float & local_alpha, float local_tolerance, struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  FirstItr_PageRank_cuda(0, ctx->nowned, local_alpha, local_tolerance, ctx);
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
  PageRank <<<blocks, __tb_PageRank>>>(ctx->gg, ctx->residual.is_updated.gpu_rd_ptr(), ctx->nowned, __begin, __end, local_alpha, local_tolerance, ctx->nout.data.gpu_wr_ptr(), ctx->residual.data.gpu_wr_ptr(), ctx->value.data.gpu_wr_ptr(), _rv);
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
