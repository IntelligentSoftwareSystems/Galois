/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "thread_work.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ tb_lb=False $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ dyn_lb=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
struct ThreadWork t_work;
bool enable_lb = true;
#include "pagerank_pull_cuda.cuh"
static const int __tb_PageRank = TB_SIZE;
static const int __tb_InitializeGraph = TB_SIZE;
__global__ void ResetGraph(CSRGraph graph, unsigned int __begin, unsigned int __end, const float  local_alpha, float * p_delta, uint32_t * p_nout, float * p_residual, float * p_value)
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
      p_value[src]    = 0;
      p_nout[src]     = 0;
      p_delta[src]    = 0;
      p_residual[src] = local_alpha;
    }
  }
  // FP: "10 -> 11;
}
__global__ void InitializeGraph_TB_LB(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_nout, DynamicBitset& bitset_nout, int * thread_prefix_work_wl, unsigned int num_items, PipeContextT<Worklist2> thread_src_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  __shared__ unsigned int total_work;
  __shared__ unsigned block_start_src_index;
  __shared__ unsigned block_end_src_index;
  unsigned my_work;
  unsigned src;
  unsigned int offset;
  unsigned int current_work;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  unsigned blockdim_x = BLOCK_DIM_X;
  // FP: "3 -> 4;
  // FP: "4 -> 5;
  // FP: "5 -> 6;
  // FP: "6 -> 7;
  // FP: "7 -> 8;
  // FP: "8 -> 9;
  // FP: "9 -> 10;
  total_work = thread_prefix_work_wl[num_items - 1];
  // FP: "10 -> 11;
  my_work = ceilf((float)(total_work) / (float) nthreads);
  // FP: "11 -> 12;

  // FP: "12 -> 13;
  __syncthreads();
  // FP: "13 -> 14;

  // FP: "14 -> 15;
  if (my_work != 0)
  {
    current_work = tid;
  }
  // FP: "17 -> 18;
  for (unsigned i =0; i < my_work; i++)
  {
    unsigned int block_start_work;
    unsigned int block_end_work;
    if (threadIdx.x == 0)
    {
      if (current_work < total_work)
      {
        block_start_work = current_work;
        block_end_work=current_work + blockdim_x - 1;
        if (block_end_work >= total_work)
        {
          block_end_work = total_work - 1;
        }
        block_start_src_index = compute_src_and_offset(0, num_items - 1,  block_start_work+1, thread_prefix_work_wl, num_items,offset);
        block_end_src_index = compute_src_and_offset(0, num_items - 1, block_end_work+1, thread_prefix_work_wl, num_items, offset);
      }
    }
    __syncthreads();

    if (current_work < total_work)
    {
      unsigned src_index;
      index_type nbr;
      src_index = compute_src_and_offset(block_start_src_index, block_end_src_index, current_work+1, thread_prefix_work_wl,num_items, offset);
      src= thread_src_wl.in_wl().dwl[src_index];
      nbr = (graph).getFirstEdge(src)+ offset;
      {
        index_type dst;
        dst = graph.getAbsDestination(nbr);
        atomicTestAdd(&p_nout[dst], (uint32_t)1);
        bitset_nout.set(dst);
      }
      current_work = current_work + nthreads;
    }
  }
  // FP: "43 -> 44;
}
__global__ void InitializeGraph(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_nout, DynamicBitset& bitset_nout, PipeContextT<Worklist2> thread_work_wl, PipeContextT<Worklist2> thread_src_wl, bool enable_lb)
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
  unsigned d_limit = DEGREE_LIMIT;
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
    int index;
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    // FP: "6 -> 7;
    bool pop  = src < __end && ((( src < (graph).nnodes )) ? true: false);
    // FP: "7 -> 8;
    if (pop)
    {
    }
    // FP: "9 -> 10;
    // FP: "12 -> 13;
    // FP: "13 -> 14;
    int threshold = TOTAL_THREADS_1D;
    // FP: "14 -> 15;
    if (pop && (graph).getOutDegree(src) >= threshold)
    {
      index = thread_work_wl.in_wl().push_range(1) ;
      thread_src_wl.in_wl().push_range(1);
      thread_work_wl.in_wl().dwl[index] = (graph).getOutDegree(src);
      thread_src_wl.in_wl().dwl[index] = src;
      pop = false;
    }
    // FP: "17 -> 18;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "18 -> 19;
    __shared__ struct { ; } _np_closure [TB_SIZE];
    // FP: "19 -> 20;
    // FP: "20 -> 21;
    if (pop)
    {
      _np.size = (graph).getOutDegree(src);
      _np.start = (graph).getFirstEdge(src);
    }
    // FP: "23 -> 24;
    // FP: "24 -> 25;
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    // FP: "25 -> 26;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    // FP: "26 -> 27;
    if (threadIdx.x == 0)
    {
      nps.tb.owner = MAX_TB_SIZE + 1;
    }
    // FP: "29 -> 30;
    __syncthreads();
    // FP: "30 -> 31;
    while (true)
    {
      // FP: "31 -> 32;
      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }
      // FP: "34 -> 35;
      __syncthreads();
      // FP: "35 -> 36;
      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {
        // FP: "36 -> 37;
        __syncthreads();
        // FP: "37 -> 38;
        break;
      }
      // FP: "39 -> 40;
      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }
      // FP: "42 -> 43;
      __syncthreads();
      // FP: "43 -> 44;
      int ns = nps.tb.start;
      int ne = nps.tb.size;
      // FP: "44 -> 45;
      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }
      // FP: "47 -> 48;
      assert(nps.tb.src < __kernel_tb_size);
      // FP: "48 -> 49;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type nbr;
        nbr = ns +_np_j;
        {
          index_type dst;
          dst = graph.getAbsDestination(nbr);
          atomicTestAdd(&p_nout[dst], (uint32_t)1);
          bitset_nout.set(dst);
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
      while (__any_sync(0xffffffff, _np.size >= _NP_CROSSOVER_WP && _np.size < _NP_CROSSOVER_TB))
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
            atomicTestAdd(&p_nout[dst], (uint32_t)1);
            bitset_nout.set(dst);
          }
        }
      }
      // FP: "79 -> 80;
      __syncthreads();
      // FP: "80 -> 81;
    }

    // FP: "81 -> 82;
    __syncthreads();
    // FP: "82 -> 83;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "83 -> 84;
    while (_np.work())
    {
      // FP: "84 -> 85;
      int _np_i =0;
      // FP: "85 -> 86;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "86 -> 87;
      __syncthreads();
      // FP: "87 -> 88;

      // FP: "88 -> 89;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type nbr;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        nbr= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          dst = graph.getAbsDestination(nbr);
          atomicTestAdd(&p_nout[dst], (uint32_t)1);
          bitset_nout.set(dst);
        }
      }
      // FP: "97 -> 98;
      _np.execute_round_done(ITSIZE);
      // FP: "98 -> 99;
      __syncthreads();
    }
    // FP: "100 -> 101;
    assert(threadIdx.x < __kernel_tb_size);
  }
  // FP: "102 -> 103;
}
__global__ void PageRank_delta(CSRGraph graph, unsigned int __begin, unsigned int __end, const float  local_alpha, float local_tolerance, float * p_delta, uint32_t * p_nout, float * p_residual, float * p_value, HGAccumulator<unsigned int> active_vertices)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  __shared__ cub::BlockReduce<unsigned int, TB_SIZE>::TempStorage active_vertices_ts;
  index_type src_end;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  active_vertices.thread_entry();
  // FP: "3 -> 4;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      p_delta[src] = 0;
      if (p_residual[src] > 0)
      {
        p_value[src] += p_residual[src];
        if (p_residual[src] > local_tolerance)
        {
          if (p_nout[src] > 0)
          {
            p_delta[src] = p_residual[src] * (1 - local_alpha) / p_nout[src];
            active_vertices.reduce( 1);
          }
        }
        p_residual[src] = 0;
      }
    }
  }
  // FP: "19 -> 20;
  active_vertices.thread_exit<cub::BlockReduce<unsigned int, TB_SIZE> >(active_vertices_ts);
  // FP: "20 -> 21;
}
__global__ void PageRank_TB_LB(CSRGraph graph, unsigned int __begin, unsigned int __end, float * p_delta, float * p_residual, DynamicBitset& bitset_residual, int * thread_prefix_work_wl, unsigned int num_items, PipeContextT<Worklist2> thread_src_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  __shared__ unsigned int total_work;
  __shared__ unsigned block_start_src_index;
  __shared__ unsigned block_end_src_index;
  unsigned my_work;
  unsigned src;
  unsigned int offset;
  unsigned int current_work;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  unsigned blockdim_x = BLOCK_DIM_X;
  // FP: "3 -> 4;
  // FP: "4 -> 5;
  // FP: "5 -> 6;
  // FP: "6 -> 7;
  // FP: "7 -> 8;
  // FP: "8 -> 9;
  // FP: "9 -> 10;
  total_work = thread_prefix_work_wl[num_items - 1];
  // FP: "10 -> 11;
  my_work = ceilf((float)(total_work) / (float) nthreads);
  // FP: "11 -> 12;

  // FP: "12 -> 13;
  __syncthreads();
  // FP: "13 -> 14;

  // FP: "14 -> 15;
  if (my_work != 0)
  {
    current_work = tid;
  }
  // FP: "17 -> 18;
  for (unsigned i =0; i < my_work; i++)
  {
    unsigned int block_start_work;
    unsigned int block_end_work;
    if (threadIdx.x == 0)
    {
      if (current_work < total_work)
      {
        block_start_work = current_work;
        block_end_work=current_work + blockdim_x - 1;
        if (block_end_work >= total_work)
        {
          block_end_work = total_work - 1;
        }
        block_start_src_index = compute_src_and_offset(0, num_items - 1,  block_start_work+1, thread_prefix_work_wl, num_items,offset);
        block_end_src_index = compute_src_and_offset(0, num_items - 1, block_end_work+1, thread_prefix_work_wl, num_items, offset);
      }
    }
    __syncthreads();

    if (current_work < total_work)
    {
      unsigned src_index;
      index_type nbr;
      src_index = compute_src_and_offset(block_start_src_index, block_end_src_index, current_work+1, thread_prefix_work_wl,num_items, offset);
      src= thread_src_wl.in_wl().dwl[src_index];
      nbr = (graph).getFirstEdge(src)+ offset;
      {
        index_type dst;
        dst = graph.getAbsDestination(nbr);
        if (p_delta[dst] > 0)
        {
          atomicTestAdd(&p_residual[src], p_delta[dst]);
          bitset_residual.set(src);
        }
      }
      current_work = current_work + nthreads;
    }
  }
  // FP: "45 -> 46;
}
__global__ void PageRank(CSRGraph graph, unsigned int __begin, unsigned int __end, float * p_delta, float * p_residual, DynamicBitset& bitset_residual, PipeContextT<Worklist2> thread_work_wl, PipeContextT<Worklist2> thread_src_wl, bool enable_lb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_PageRank;
  index_type src_end;
  index_type src_rup;
  // FP: "1 -> 2;
  const int _NP_CROSSOVER_WP = 32;
  const int _NP_CROSSOVER_TB = __kernel_tb_size;
  // FP: "2 -> 3;
  const int BLKSIZE = __kernel_tb_size;
  const int ITSIZE = BLKSIZE * 8;
  unsigned d_limit = DEGREE_LIMIT;
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
    int index;
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    // FP: "6 -> 7;
    bool pop  = src < __end && ((( src < (graph).nnodes )) ? true: false);
    // FP: "7 -> 8;
    if (pop)
    {
    }
    // FP: "9 -> 10;
    // FP: "12 -> 13;
    // FP: "13 -> 14;
    int threshold = TOTAL_THREADS_1D;
    // FP: "14 -> 15;
    if (pop && (graph).getOutDegree(src) >= threshold)
    {
      index = thread_work_wl.in_wl().push_range(1) ;
      thread_src_wl.in_wl().push_range(1);
      thread_work_wl.in_wl().dwl[index] = (graph).getOutDegree(src);
      thread_src_wl.in_wl().dwl[index] = src;
      pop = false;
    }
    // FP: "17 -> 18;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "18 -> 19;
    __shared__ struct { index_type src; } _np_closure [TB_SIZE];
    // FP: "19 -> 20;
    _np_closure[threadIdx.x].src = src;
    // FP: "20 -> 21;
    if (pop)
    {
      _np.size = (graph).getOutDegree(src);
      _np.start = (graph).getFirstEdge(src);
    }
    // FP: "23 -> 24;
    // FP: "24 -> 25;
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    // FP: "25 -> 26;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    // FP: "26 -> 27;
    if (threadIdx.x == 0)
    {
      nps.tb.owner = MAX_TB_SIZE + 1;
    }
    // FP: "29 -> 30;
    __syncthreads();
    // FP: "30 -> 31;
    while (true)
    {
      // FP: "31 -> 32;
      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }
      // FP: "34 -> 35;
      __syncthreads();
      // FP: "35 -> 36;
      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {
        // FP: "36 -> 37;
        __syncthreads();
        // FP: "37 -> 38;
        break;
      }
      // FP: "39 -> 40;
      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }
      // FP: "42 -> 43;
      __syncthreads();
      // FP: "43 -> 44;
      int ns = nps.tb.start;
      int ne = nps.tb.size;
      // FP: "44 -> 45;
      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }
      // FP: "47 -> 48;
      assert(nps.tb.src < __kernel_tb_size);
      src = _np_closure[nps.tb.src].src;
      // FP: "48 -> 49;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type nbr;
        nbr = ns +_np_j;
        {
          index_type dst;
          dst = graph.getAbsDestination(nbr);
          if (p_delta[dst] > 0)
          {
            atomicTestAdd(&p_residual[src], p_delta[dst]);
            bitset_residual.set(src);
          }
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
      while (__any_sync(0xffffffff, _np.size >= _NP_CROSSOVER_WP && _np.size < _NP_CROSSOVER_TB))
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
            dst = graph.getAbsDestination(nbr);
            if (p_delta[dst] > 0)
            {
              atomicTestAdd(&p_residual[src], p_delta[dst]);
              bitset_residual.set(src);
            }
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
        src = _np_closure[nps.fg.src[_np_i]].src;
        nbr= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          dst = graph.getAbsDestination(nbr);
          if (p_delta[dst] > 0)
          {
            atomicTestAdd(&p_residual[src], p_delta[dst]);
            bitset_residual.set(src);
          }
        }
      }
      // FP: "103 -> 104;
      _np.execute_round_done(ITSIZE);
      // FP: "104 -> 105;
      __syncthreads();
    }
    // FP: "106 -> 107;
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
  }
  // FP: "108 -> 109;
}
__global__ void PageRankSanity(CSRGraph graph, unsigned int __begin, unsigned int __end, float local_tolerance, float * p_residual, float * p_value, HGAccumulator<uint64_t> DGAccumulator_residual_over_tolerance, HGAccumulator<float> DGAccumulator_sum, HGAccumulator<float> DGAccumulator_sum_residual, HGReduceMax<float> max_residual, HGReduceMax<float> max_value, HGReduceMin<float> min_residual, HGReduceMin<float> min_value)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  __shared__ cub::BlockReduce<uint64_t, TB_SIZE>::TempStorage DGAccumulator_residual_over_tolerance_ts;
  __shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage DGAccumulator_sum_ts;
  __shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage DGAccumulator_sum_residual_ts;
  __shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage max_residual_ts;
  __shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage max_value_ts;
  __shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage min_residual_ts;
  __shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage min_value_ts;
  index_type src_end;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  DGAccumulator_residual_over_tolerance.thread_entry();
  // FP: "3 -> 4;
  // FP: "4 -> 5;
  DGAccumulator_sum.thread_entry();
  // FP: "5 -> 6;
  // FP: "6 -> 7;
  DGAccumulator_sum_residual.thread_entry();
  // FP: "7 -> 8;
  // FP: "8 -> 9;
  max_residual.thread_entry();
  // FP: "9 -> 10;
  // FP: "10 -> 11;
  max_value.thread_entry();
  // FP: "11 -> 12;
  // FP: "12 -> 13;
  min_residual.thread_entry();
  // FP: "13 -> 14;
  // FP: "14 -> 15;
  min_value.thread_entry();
  // FP: "15 -> 16;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      max_value.reduce(p_value[src]);
      min_value.reduce(p_value[src]);
      max_residual.reduce(p_residual[src]);
      min_residual.reduce(p_residual[src]);
      DGAccumulator_sum.reduce( p_value[src]);
      DGAccumulator_sum.reduce( p_residual[src]);
      if (p_residual[src] > local_tolerance)
      {
        DGAccumulator_residual_over_tolerance.reduce( 1);
      }
    }
  }
  // FP: "29 -> 30;
  DGAccumulator_residual_over_tolerance.thread_exit<cub::BlockReduce<uint64_t, TB_SIZE> >(DGAccumulator_residual_over_tolerance_ts);
  // FP: "30 -> 31;
  DGAccumulator_sum.thread_exit<cub::BlockReduce<float, TB_SIZE> >(DGAccumulator_sum_ts);
  // FP: "31 -> 32;
  DGAccumulator_sum_residual.thread_exit<cub::BlockReduce<float, TB_SIZE> >(DGAccumulator_sum_residual_ts);
  // FP: "32 -> 33;
  max_residual.thread_exit<cub::BlockReduce<float, TB_SIZE> >(max_residual_ts);
  // FP: "33 -> 34;
  max_value.thread_exit<cub::BlockReduce<float, TB_SIZE> >(max_value_ts);
  // FP: "34 -> 35;
  min_residual.thread_exit<cub::BlockReduce<float, TB_SIZE> >(min_residual_ts);
  // FP: "35 -> 36;
  min_value.thread_exit<cub::BlockReduce<float, TB_SIZE> >(min_value_ts);
  // FP: "36 -> 37;
}
void ResetGraph_cuda(unsigned int  __begin, unsigned int  __end, const float & local_alpha, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  ResetGraph <<<blocks, threads>>>(ctx->gg, __begin, __end, local_alpha, ctx->delta.data.gpu_wr_ptr(), ctx->nout.data.gpu_wr_ptr(), ctx->residual.data.gpu_wr_ptr(), ctx->value.data.gpu_wr_ptr());
  cudaDeviceSynchronize();
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void ResetGraph_allNodes_cuda(const float & local_alpha, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  ResetGraph_cuda(0, ctx->gg.nnodes, local_alpha, ctx);
  // FP: "2 -> 3;
}
void ResetGraph_masterNodes_cuda(const float & local_alpha, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  ResetGraph_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_alpha, ctx);
  // FP: "2 -> 3;
}
void ResetGraph_nodesWithEdges_cuda(const float & local_alpha, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  ResetGraph_cuda(0, ctx->numNodesWithEdges, local_alpha, ctx);
  // FP: "2 -> 3;
}
void InitializeGraph_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context*  ctx)
{
  t_work.init_thread_work(ctx->gg.nnodes);
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  InitializeGraph <<<blocks, __tb_InitializeGraph>>>(ctx->gg, __begin, __end, ctx->nout.data.gpu_wr_ptr(), *(ctx->nout.is_updated.gpu_rd_ptr()), t_work.thread_work_wl, t_work.thread_src_wl, enable_lb);
  cudaDeviceSynchronize();
  if (enable_lb)
  {
    int num_items = t_work.thread_work_wl.in_wl().nitems();
    if (num_items != 0)
    {
      t_work.compute_prefix_sum();
      cudaDeviceSynchronize();
      InitializeGraph_TB_LB <<<blocks, __tb_InitializeGraph>>>(ctx->gg, __begin, __end, ctx->nout.data.gpu_wr_ptr(), *(ctx->nout.is_updated.gpu_rd_ptr()), t_work.thread_prefix_work_wl.gpu_wr_ptr(), num_items, t_work.thread_src_wl);
      cudaDeviceSynchronize();
      t_work.reset_thread_work();
    }
  }
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void InitializeGraph_allNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  InitializeGraph_cuda(0, ctx->gg.nnodes, ctx);
  // FP: "2 -> 3;
}
void InitializeGraph_masterNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  InitializeGraph_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx);
  // FP: "2 -> 3;
}
void InitializeGraph_nodesWithEdges_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  InitializeGraph_cuda(0, ctx->numNodesWithEdges, ctx);
  // FP: "2 -> 3;
}
void PageRank_delta_cuda(unsigned int  __begin, unsigned int  __end, unsigned int & active_vertices, const float & local_alpha, float local_tolerance, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  HGAccumulator<unsigned int> _active_vertices;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  Shared<unsigned int> active_verticesval  = Shared<unsigned int>(1);
  // FP: "5 -> 6;
  // FP: "6 -> 7;
  *(active_verticesval.cpu_wr_ptr()) = 0;
  // FP: "7 -> 8;
  _active_vertices.rv = active_verticesval.gpu_wr_ptr();
  // FP: "8 -> 9;
  PageRank_delta <<<blocks, threads>>>(ctx->gg, __begin, __end, local_alpha, local_tolerance, ctx->delta.data.gpu_wr_ptr(), ctx->nout.data.gpu_wr_ptr(), ctx->residual.data.gpu_wr_ptr(), ctx->value.data.gpu_wr_ptr(), _active_vertices);
  cudaDeviceSynchronize();
  // FP: "9 -> 10;
  check_cuda_kernel;
  // FP: "10 -> 11;
  active_vertices = *(active_verticesval.cpu_rd_ptr());
  // FP: "11 -> 12;
}
void PageRank_delta_allNodes_cuda(unsigned int & active_vertices, const float & local_alpha, float local_tolerance, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  PageRank_delta_cuda(0, ctx->gg.nnodes, active_vertices, local_alpha, local_tolerance, ctx);
  // FP: "2 -> 3;
}
void PageRank_delta_masterNodes_cuda(unsigned int & active_vertices, const float & local_alpha, float local_tolerance, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  PageRank_delta_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, active_vertices, local_alpha, local_tolerance, ctx);
  // FP: "2 -> 3;
}
void PageRank_delta_nodesWithEdges_cuda(unsigned int & active_vertices, const float & local_alpha, float local_tolerance, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  PageRank_delta_cuda(0, ctx->numNodesWithEdges, active_vertices, local_alpha, local_tolerance, ctx);
  // FP: "2 -> 3;
}
void PageRank_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  PageRank <<<blocks, __tb_PageRank>>>(ctx->gg, __begin, __end, ctx->delta.data.gpu_wr_ptr(), ctx->residual.data.gpu_wr_ptr(), *(ctx->residual.is_updated.gpu_rd_ptr()), t_work.thread_work_wl, t_work.thread_src_wl, enable_lb);
  cudaDeviceSynchronize();
  if (enable_lb)
  {
    int num_items = t_work.thread_work_wl.in_wl().nitems();
    if (num_items != 0)
    {
      t_work.compute_prefix_sum();
      cudaDeviceSynchronize();
      PageRank_TB_LB <<<blocks, __tb_PageRank>>>(ctx->gg, __begin, __end, ctx->delta.data.gpu_wr_ptr(), ctx->residual.data.gpu_wr_ptr(), *(ctx->residual.is_updated.gpu_rd_ptr()), t_work.thread_prefix_work_wl.gpu_wr_ptr(), num_items, t_work.thread_src_wl);
      cudaDeviceSynchronize();
      t_work.reset_thread_work();
    }
  }
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void PageRank_allNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  PageRank_cuda(0, ctx->gg.nnodes, ctx);
  // FP: "2 -> 3;
}
void PageRank_masterNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  PageRank_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx);
  // FP: "2 -> 3;
}
void PageRank_nodesWithEdges_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  PageRank_cuda(0, ctx->numNodesWithEdges, ctx);
  // FP: "2 -> 3;
}
void PageRankSanity_cuda(unsigned int  __begin, unsigned int  __end, uint64_t & DGAccumulator_residual_over_tolerance, float & DGAccumulator_sum, float & DGAccumulator_sum_residual, float & max_residual, float & max_value, float & min_residual, float & min_value, float local_tolerance, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  HGAccumulator<uint64_t> _DGAccumulator_residual_over_tolerance;
  HGAccumulator<float> _DGAccumulator_sum;
  HGAccumulator<float> _DGAccumulator_sum_residual;
  HGReduceMax<float> _max_residual;
  HGReduceMax<float> _max_value;
  HGReduceMin<float> _min_residual;
  HGReduceMin<float> _min_value;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  Shared<uint64_t> DGAccumulator_residual_over_toleranceval  = Shared<uint64_t>(1);
  // FP: "5 -> 6;
  // FP: "6 -> 7;
  *(DGAccumulator_residual_over_toleranceval.cpu_wr_ptr()) = 0;
  // FP: "7 -> 8;
  _DGAccumulator_residual_over_tolerance.rv = DGAccumulator_residual_over_toleranceval.gpu_wr_ptr();
  // FP: "8 -> 9;
  Shared<float> DGAccumulator_sumval  = Shared<float>(1);
  // FP: "9 -> 10;
  // FP: "10 -> 11;
  *(DGAccumulator_sumval.cpu_wr_ptr()) = 0;
  // FP: "11 -> 12;
  _DGAccumulator_sum.rv = DGAccumulator_sumval.gpu_wr_ptr();
  // FP: "12 -> 13;
  Shared<float> DGAccumulator_sum_residualval  = Shared<float>(1);
  // FP: "13 -> 14;
  // FP: "14 -> 15;
  *(DGAccumulator_sum_residualval.cpu_wr_ptr()) = 0;
  // FP: "15 -> 16;
  _DGAccumulator_sum_residual.rv = DGAccumulator_sum_residualval.gpu_wr_ptr();
  // FP: "16 -> 17;
  Shared<float> max_residualval  = Shared<float>(1);
  // FP: "17 -> 18;
  // FP: "18 -> 19;
  *(max_residualval.cpu_wr_ptr()) = 0;
  // FP: "19 -> 20;
  _max_residual.rv = max_residualval.gpu_wr_ptr();
  // FP: "20 -> 21;
  Shared<float> max_valueval  = Shared<float>(1);
  // FP: "21 -> 22;
  // FP: "22 -> 23;
  *(max_valueval.cpu_wr_ptr()) = 0;
  // FP: "23 -> 24;
  _max_value.rv = max_valueval.gpu_wr_ptr();
  // FP: "24 -> 25;
  Shared<float> min_residualval  = Shared<float>(1);
  // FP: "25 -> 26;
  // FP: "26 -> 27;
  *(min_residualval.cpu_wr_ptr()) = 1073741823;
  // FP: "27 -> 28;
  _min_residual.rv = min_residualval.gpu_wr_ptr();
  // FP: "28 -> 29;
  Shared<float> min_valueval  = Shared<float>(1);
  // FP: "29 -> 30;
  // FP: "30 -> 31;
  *(min_valueval.cpu_wr_ptr()) = 1073741823;
  // FP: "31 -> 32;
  _min_value.rv = min_valueval.gpu_wr_ptr();
  // FP: "32 -> 33;
  PageRankSanity <<<blocks, threads>>>(ctx->gg, __begin, __end, local_tolerance, ctx->residual.data.gpu_wr_ptr(), ctx->value.data.gpu_wr_ptr(), _DGAccumulator_residual_over_tolerance, _DGAccumulator_sum, _DGAccumulator_sum_residual, _max_residual, _max_value, _min_residual, _min_value);
  cudaDeviceSynchronize();
  // FP: "33 -> 34;
  check_cuda_kernel;
  // FP: "34 -> 35;
  DGAccumulator_residual_over_tolerance = *(DGAccumulator_residual_over_toleranceval.cpu_rd_ptr());
  // FP: "35 -> 36;
  DGAccumulator_sum = *(DGAccumulator_sumval.cpu_rd_ptr());
  // FP: "36 -> 37;
  DGAccumulator_sum_residual = *(DGAccumulator_sum_residualval.cpu_rd_ptr());
  // FP: "37 -> 38;
  max_residual = *(max_residualval.cpu_rd_ptr());
  // FP: "38 -> 39;
  max_value = *(max_valueval.cpu_rd_ptr());
  // FP: "39 -> 40;
  min_residual = *(min_residualval.cpu_rd_ptr());
  // FP: "40 -> 41;
  min_value = *(min_valueval.cpu_rd_ptr());
  // FP: "41 -> 42;
}
void PageRankSanity_allNodes_cuda(uint64_t & DGAccumulator_residual_over_tolerance, float & DGAccumulator_sum, float & DGAccumulator_sum_residual, float & max_residual, float & max_value, float & min_residual, float & min_value, float local_tolerance, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  PageRankSanity_cuda(0, ctx->gg.nnodes, DGAccumulator_residual_over_tolerance, DGAccumulator_sum, DGAccumulator_sum_residual, max_residual, max_value, min_residual, min_value, local_tolerance, ctx);
  // FP: "2 -> 3;
}
void PageRankSanity_masterNodes_cuda(uint64_t & DGAccumulator_residual_over_tolerance, float & DGAccumulator_sum, float & DGAccumulator_sum_residual, float & max_residual, float & max_value, float & min_residual, float & min_value, float local_tolerance, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  PageRankSanity_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, DGAccumulator_residual_over_tolerance, DGAccumulator_sum, DGAccumulator_sum_residual, max_residual, max_value, min_residual, min_value, local_tolerance, ctx);
  // FP: "2 -> 3;
}
void PageRankSanity_nodesWithEdges_cuda(uint64_t & DGAccumulator_residual_over_tolerance, float & DGAccumulator_sum, float & DGAccumulator_sum_residual, float & max_residual, float & max_value, float & min_residual, float & min_value, float local_tolerance, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  PageRankSanity_cuda(0, ctx->numNodesWithEdges, DGAccumulator_residual_over_tolerance, DGAccumulator_sum, DGAccumulator_sum_residual, max_residual, max_value, min_residual, min_value, local_tolerance, ctx);
  // FP: "2 -> 3;
}
