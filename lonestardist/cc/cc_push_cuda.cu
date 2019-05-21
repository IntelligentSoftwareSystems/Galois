/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "thread_work.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ tb_lb=True $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
struct ThreadWork t_work;
bool enable_lb = true;
#include "kernels/reduce.cuh"
#include "cc_push_cuda.cuh"
static const int __tb_FirstItr_ConnectedComp = TB_SIZE;
static const int __tb_ConnectedComp = TB_SIZE;
__global__ void InitializeGraph(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_comp_current, uint32_t * p_comp_old)
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
      p_comp_old[src]     = graph.node_data[src];
    }
  }
  // FP: "8 -> 9;
}
__global__ void FirstItr_ConnectedComp_TB_LB(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_comp_current, uint32_t * p_comp_old, DynamicBitset& bitset_comp_current, int * thread_prefix_work_wl, unsigned int num_items, PipeContextT<Worklist2> thread_src_wl)
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
      index_type jj;
      src_index = compute_src_and_offset(block_start_src_index, block_end_src_index, current_work+1, thread_prefix_work_wl,num_items, offset);
      src= thread_src_wl.in_wl().dwl[src_index];
      jj = (graph).getFirstEdge(src)+ offset;
      {
        index_type dst;
        uint32_t new_dist;
        uint32_t old_dist;
        dst = graph.getAbsDestination(jj);
        new_dist = p_comp_current[src];
        old_dist = atomicTestMin(&p_comp_current[dst], new_dist);
        if (old_dist > new_dist)
        {
          bitset_comp_current.set(dst);
        }
      }
      current_work = current_work + nthreads;
    }
  }
  // FP: "48 -> 49;
}
__global__ void Inspect_FirstItr_ConnectedComp(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_comp_current, uint32_t * p_comp_old, DynamicBitset& bitset_comp_current, PipeContextT<Worklist2> thread_work_wl, PipeContextT<Worklist2> thread_src_wl, bool enable_lb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  // FP: "1 -> 2;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    int index;
    bool pop  = src < __end && ((( src < (graph).nnodes ) && ( (graph).getOutDegree(src) >= DEGREE_LIMIT)) ? true: false);
    if (pop)
    {
      p_comp_old[src]  = p_comp_current[src];
    }
    if (!pop)
    {
      continue;
    }
    if (pop)
    {
      index = thread_work_wl.in_wl().push_range(1) ;
      thread_src_wl.in_wl().push_range(1);
      thread_work_wl.in_wl().dwl[index] = (graph).getOutDegree(src);
      thread_src_wl.in_wl().dwl[index] = src;
    }
  }
  // FP: "14 -> 15;
}
__global__ void FirstItr_ConnectedComp(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_comp_current, uint32_t * p_comp_old, DynamicBitset& bitset_comp_current, bool enable_lb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_FirstItr_ConnectedComp;
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
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    // FP: "6 -> 7;
    bool pop  = src < __end && ((( src < (graph).nnodes ) && ( (graph).getOutDegree(src) < DEGREE_LIMIT)) ? true: false);
    // FP: "7 -> 8;
    if (pop)
    {
      p_comp_old[src]  = p_comp_current[src];
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
        index_type jj;
        jj = ns +_np_j;
        {
          index_type dst;
          uint32_t new_dist;
          uint32_t old_dist;
          dst = graph.getAbsDestination(jj);
          new_dist = p_comp_current[src];
          old_dist = atomicTestMin(&p_comp_current[dst], new_dist);
          if (old_dist > new_dist)
          {
            bitset_comp_current.set(dst);
          }
        }
      }
      // FP: "57 -> 58;
      __syncthreads();
    }
    // FP: "59 -> 60;

    // FP: "60 -> 61;
    {
      const int warpid = threadIdx.x / 32;
      // FP: "61 -> 62;
      const int _np_laneid = cub::LaneId();
      // FP: "62 -> 63;
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
            uint32_t new_dist;
            uint32_t old_dist;
            dst = graph.getAbsDestination(jj);
            new_dist = p_comp_current[src];
            old_dist = atomicTestMin(&p_comp_current[dst], new_dist);
            if (old_dist > new_dist)
            {
              bitset_comp_current.set(dst);
            }
          }
        }
      }
      // FP: "85 -> 86;
      __syncthreads();
      // FP: "86 -> 87;
    }

    // FP: "87 -> 88;
    __syncthreads();
    // FP: "88 -> 89;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "89 -> 90;
    while (_np.work())
    {
      // FP: "90 -> 91;
      int _np_i =0;
      // FP: "91 -> 92;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "92 -> 93;
      __syncthreads();
      // FP: "93 -> 94;

      // FP: "94 -> 95;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type jj;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        jj= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          uint32_t new_dist;
          uint32_t old_dist;
          dst = graph.getAbsDestination(jj);
          new_dist = p_comp_current[src];
          old_dist = atomicTestMin(&p_comp_current[dst], new_dist);
          if (old_dist > new_dist)
          {
            bitset_comp_current.set(dst);
          }
        }
      }
      // FP: "108 -> 109;
      _np.execute_round_done(ITSIZE);
      // FP: "109 -> 110;
      __syncthreads();
    }
    // FP: "111 -> 112;
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
  }
  // FP: "113 -> 114;
}
__global__ void ConnectedComp_TB_LB(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_comp_current, uint32_t * p_comp_old, DynamicBitset& bitset_comp_current, HGAccumulator<unsigned int> active_vertices, int * thread_prefix_work_wl, unsigned int num_items, PipeContextT<Worklist2> thread_src_wl)
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
      index_type jj;
      src_index = compute_src_and_offset(block_start_src_index, block_end_src_index, current_work+1, thread_prefix_work_wl,num_items, offset);
      src= thread_src_wl.in_wl().dwl[src_index];
      jj = (graph).getFirstEdge(src)+ offset;
      {
        index_type dst;
        uint32_t new_dist;
        uint32_t old_dist;
        active_vertices.reduce( 1);
        dst = graph.getAbsDestination(jj);
        new_dist = p_comp_current[src];
        old_dist = atomicTestMin(&p_comp_current[dst], new_dist);
        if (old_dist > new_dist)
        {
          bitset_comp_current.set(dst);
        }
      }
      current_work = current_work + nthreads;
    }
  }
  // FP: "49 -> 50;
}
__global__ void Inspect_ConnectedComp(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_comp_current, uint32_t * p_comp_old, DynamicBitset& bitset_comp_current, HGAccumulator<unsigned int> active_vertices, PipeContextT<Worklist2> thread_work_wl, PipeContextT<Worklist2> thread_src_wl, bool enable_lb)
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
    int index;
    bool pop  = src < __end && ((( src < (graph).nnodes ) && ( (graph).getOutDegree(src) >= DEGREE_LIMIT)) ? true: false);
    if (pop)
    {
      if (p_comp_old[src] > p_comp_current[src])
      {
        p_comp_old[src] = p_comp_current[src];
      }
      else
      {
        pop = false;
      }
    }
    if (!pop)
    {
      continue;
    }
    if (pop)
    {
      index = thread_work_wl.in_wl().push_range(1) ;
      thread_src_wl.in_wl().push_range(1);
      thread_work_wl.in_wl().dwl[index] = (graph).getOutDegree(src);
      thread_src_wl.in_wl().dwl[index] = src;
    }
  }
  // FP: "19 -> 20;
  active_vertices.thread_exit<cub::BlockReduce<unsigned int, TB_SIZE> >(active_vertices_ts);
  // FP: "20 -> 21;
}
__global__ void ConnectedComp(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_comp_current, uint32_t * p_comp_old, DynamicBitset& bitset_comp_current, HGAccumulator<unsigned int> active_vertices, bool enable_lb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_ConnectedComp;
  __shared__ cub::BlockReduce<unsigned int, TB_SIZE>::TempStorage active_vertices_ts;
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
  // FP: "6 -> 7;
  active_vertices.thread_entry();
  // FP: "7 -> 8;
  src_end = __end;
  src_rup = ((__begin) + roundup(((__end) - (__begin)), (blockDim.x)));
  for (index_type src = __begin + tid; src < src_rup; src += nthreads)
  {
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    // FP: "8 -> 9;
    bool pop  = src < __end && ((( src < (graph).nnodes ) && ( (graph).getOutDegree(src) < DEGREE_LIMIT)) ? true: false);
    // FP: "9 -> 10;
    if (pop)
    {
      if (p_comp_old[src] > p_comp_current[src])
      {
        p_comp_old[src] = p_comp_current[src];
      }
      else
      {
        pop = false;
      }
    }
    // FP: "14 -> 15;
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
        index_type jj;
        jj = ns +_np_j;
        {
          index_type dst;
          uint32_t new_dist;
          uint32_t old_dist;
          active_vertices.reduce( 1);
          dst = graph.getAbsDestination(jj);
          new_dist = p_comp_current[src];
          old_dist = atomicTestMin(&p_comp_current[dst], new_dist);
          if (old_dist > new_dist)
          {
            bitset_comp_current.set(dst);
          }
        }
      }
      // FP: "62 -> 63;
      __syncthreads();
    }
    // FP: "64 -> 65;

    // FP: "65 -> 66;
    {
      const int warpid = threadIdx.x / 32;
      // FP: "66 -> 67;
      const int _np_laneid = cub::LaneId();
      // FP: "67 -> 68;
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
            uint32_t new_dist;
            uint32_t old_dist;
            active_vertices.reduce( 1);
            dst = graph.getAbsDestination(jj);
            new_dist = p_comp_current[src];
            old_dist = atomicTestMin(&p_comp_current[dst], new_dist);
            if (old_dist > new_dist)
            {
              bitset_comp_current.set(dst);
            }
          }
        }
      }
      // FP: "91 -> 92;
      __syncthreads();
      // FP: "92 -> 93;
    }

    // FP: "93 -> 94;
    __syncthreads();
    // FP: "94 -> 95;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "95 -> 96;
    while (_np.work())
    {
      // FP: "96 -> 97;
      int _np_i =0;
      // FP: "97 -> 98;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "98 -> 99;
      __syncthreads();
      // FP: "99 -> 100;

      // FP: "100 -> 101;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type jj;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        jj= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          uint32_t new_dist;
          uint32_t old_dist;
          active_vertices.reduce( 1);
          dst = graph.getAbsDestination(jj);
          new_dist = p_comp_current[src];
          old_dist = atomicTestMin(&p_comp_current[dst], new_dist);
          if (old_dist > new_dist)
          {
            bitset_comp_current.set(dst);
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
  // FP: "121 -> 122;
  active_vertices.thread_exit<cub::BlockReduce<unsigned int, TB_SIZE> >(active_vertices_ts);
  // FP: "122 -> 123;
}
__global__ void ConnectedCompSanityCheck(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_comp_current, HGAccumulator<uint64_t> active_vertices)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  __shared__ cub::BlockReduce<uint64_t, TB_SIZE>::TempStorage active_vertices_ts;
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
      if (p_comp_current[src] == graph.node_data[src])
      {
        active_vertices.reduce( 1);
      }
    }
  }
  // FP: "11 -> 12;
  active_vertices.thread_exit<cub::BlockReduce<uint64_t, TB_SIZE> >(active_vertices_ts);
  // FP: "12 -> 13;
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
  InitializeGraph <<<blocks, threads>>>(ctx->gg, __begin, __end, ctx->comp_current.data.gpu_wr_ptr(), ctx->comp_old.data.gpu_wr_ptr());
  cudaDeviceSynchronize();
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
void FirstItr_ConnectedComp_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  if (enable_lb)
  {
    t_work.reset_thread_work();
    Inspect_FirstItr_ConnectedComp <<<blocks, __tb_FirstItr_ConnectedComp>>>(ctx->gg, __begin, __end, ctx->comp_current.data.gpu_wr_ptr(), ctx->comp_old.data.gpu_wr_ptr(), *(ctx->comp_current.is_updated.gpu_rd_ptr()), t_work.thread_work_wl, t_work.thread_src_wl, enable_lb);
    cudaDeviceSynchronize();
    int num_items = t_work.thread_work_wl.in_wl().nitems();
    if (num_items != 0)
    {
      t_work.compute_prefix_sum();
      cudaDeviceSynchronize();
      FirstItr_ConnectedComp_TB_LB <<<blocks, __tb_FirstItr_ConnectedComp>>>(ctx->gg, __begin, __end, ctx->comp_current.data.gpu_wr_ptr(), ctx->comp_old.data.gpu_wr_ptr(), *(ctx->comp_current.is_updated.gpu_rd_ptr()), t_work.thread_prefix_work_wl.gpu_wr_ptr(), num_items, t_work.thread_src_wl);
      cudaDeviceSynchronize();
    }
  }
  FirstItr_ConnectedComp <<<blocks, __tb_FirstItr_ConnectedComp>>>(ctx->gg, __begin, __end, ctx->comp_current.data.gpu_wr_ptr(), ctx->comp_old.data.gpu_wr_ptr(), *(ctx->comp_current.is_updated.gpu_rd_ptr()), enable_lb);
  cudaDeviceSynchronize();
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void FirstItr_ConnectedComp_allNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  FirstItr_ConnectedComp_cuda(0, ctx->gg.nnodes, ctx);
  // FP: "2 -> 3;
}
void FirstItr_ConnectedComp_masterNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  FirstItr_ConnectedComp_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx);
  // FP: "2 -> 3;
}
void FirstItr_ConnectedComp_nodesWithEdges_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  FirstItr_ConnectedComp_cuda(0, ctx->numNodesWithEdges, ctx);
  // FP: "2 -> 3;
}
void ConnectedComp_cuda(unsigned int  __begin, unsigned int  __end, unsigned int & active_vertices, struct CUDA_Context*  ctx)
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
  if (enable_lb)
  {
    t_work.reset_thread_work();
    Inspect_ConnectedComp <<<blocks, __tb_ConnectedComp>>>(ctx->gg, __begin, __end, ctx->comp_current.data.gpu_wr_ptr(), ctx->comp_old.data.gpu_wr_ptr(), *(ctx->comp_current.is_updated.gpu_rd_ptr()), _active_vertices, t_work.thread_work_wl, t_work.thread_src_wl, enable_lb);
    cudaDeviceSynchronize();
    int num_items = t_work.thread_work_wl.in_wl().nitems();
    if (num_items != 0)
    {
      t_work.compute_prefix_sum();
      cudaDeviceSynchronize();
      ConnectedComp_TB_LB <<<blocks, __tb_ConnectedComp>>>(ctx->gg, __begin, __end, ctx->comp_current.data.gpu_wr_ptr(), ctx->comp_old.data.gpu_wr_ptr(), *(ctx->comp_current.is_updated.gpu_rd_ptr()), _active_vertices, t_work.thread_prefix_work_wl.gpu_wr_ptr(), num_items, t_work.thread_src_wl);
      cudaDeviceSynchronize();
    }
  }
  ConnectedComp <<<blocks, __tb_ConnectedComp>>>(ctx->gg, __begin, __end, ctx->comp_current.data.gpu_wr_ptr(), ctx->comp_old.data.gpu_wr_ptr(), *(ctx->comp_current.is_updated.gpu_rd_ptr()), _active_vertices, enable_lb);
  cudaDeviceSynchronize();
  // FP: "9 -> 10;
  check_cuda_kernel;
  // FP: "10 -> 11;
  active_vertices = *(active_verticesval.cpu_rd_ptr());
  // FP: "11 -> 12;
}
void ConnectedComp_allNodes_cuda(unsigned int & active_vertices, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  ConnectedComp_cuda(0, ctx->gg.nnodes, active_vertices, ctx);
  // FP: "2 -> 3;
}
void ConnectedComp_masterNodes_cuda(unsigned int & active_vertices, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  ConnectedComp_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, active_vertices, ctx);
  // FP: "2 -> 3;
}
void ConnectedComp_nodesWithEdges_cuda(unsigned int & active_vertices, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  ConnectedComp_cuda(0, ctx->numNodesWithEdges, active_vertices, ctx);
  // FP: "2 -> 3;
}
void ConnectedCompSanityCheck_cuda(unsigned int  __begin, unsigned int  __end, uint64_t & active_vertices, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  HGAccumulator<uint64_t> _active_vertices;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  Shared<uint64_t> active_verticesval  = Shared<uint64_t>(1);
  // FP: "5 -> 6;
  // FP: "6 -> 7;
  *(active_verticesval.cpu_wr_ptr()) = 0;
  // FP: "7 -> 8;
  _active_vertices.rv = active_verticesval.gpu_wr_ptr();
  // FP: "8 -> 9;
  ConnectedCompSanityCheck <<<blocks, threads>>>(ctx->gg, __begin, __end, ctx->comp_current.data.gpu_wr_ptr(), _active_vertices);
  cudaDeviceSynchronize();
  // FP: "9 -> 10;
  check_cuda_kernel;
  // FP: "10 -> 11;
  active_vertices = *(active_verticesval.cpu_rd_ptr());
  // FP: "11 -> 12;
}
void ConnectedCompSanityCheck_allNodes_cuda(uint64_t & active_vertices, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  ConnectedCompSanityCheck_cuda(0, ctx->gg.nnodes, active_vertices, ctx);
  // FP: "2 -> 3;
}
void ConnectedCompSanityCheck_masterNodes_cuda(uint64_t & active_vertices, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  ConnectedCompSanityCheck_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, active_vertices, ctx);
  // FP: "2 -> 3;
}
void ConnectedCompSanityCheck_nodesWithEdges_cuda(uint64_t & active_vertices, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  ConnectedCompSanityCheck_cuda(0, ctx->numNodesWithEdges, active_vertices, ctx);
  // FP: "2 -> 3;
}