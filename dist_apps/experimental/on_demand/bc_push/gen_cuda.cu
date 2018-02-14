/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
static const int __tb_NumShortestPaths = TB_SIZE;
static const int __tb_FirstIterationSSSP = TB_SIZE;
static const int __tb_SSSP = TB_SIZE;
static const int __tb_DependencyPropagation = TB_SIZE;
static const int __tb_PredAndSucc = TB_SIZE;
__global__ void InitializeGraph(CSRGraph graph, unsigned int __begin, unsigned int __end, float * p_betweeness_centrality, float * p_dependency, uint32_t * p_num_predecessors, uint64_t * p_num_shortest_paths, uint32_t * p_num_successors, uint8_t * p_propagation_flag, uint64_t * p_to_add, float * p_to_add_float, uint32_t * p_trim)
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
      p_betweeness_centrality[src] = 0;
      p_num_shortest_paths[src] = 0;
      p_num_successors[src] = 0;
      p_num_predecessors[src] = 0;
      p_trim[src] = 0;
      p_to_add[src] = 0;
      p_to_add_float[src] = 0;
      p_dependency[src] = 0;
      p_propagation_flag[src] = false;
    }
  }
  // FP: "15 -> 16;
}
__global__ void InitializeIteration(CSRGraph graph, unsigned int __begin, unsigned int __end, const uint64_t  local_current_src_node, const uint32_t  local_infinity, uint32_t * p_current_length, float * p_dependency, uint32_t * p_num_predecessors, uint64_t * p_num_shortest_paths, uint32_t * p_num_successors, uint32_t * p_old_length, uint8_t * p_propagation_flag)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  bool is_source;
  index_type src_end;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      is_source = graph.node_data[src] == local_current_src_node;
      if (!is_source)
      {
        p_current_length[src] = local_infinity;
        p_old_length[src] = local_infinity;
        p_num_shortest_paths[src] = 0;
        p_propagation_flag[src] = false;
      }
      else
      {
        p_current_length[src] = 0;
        p_old_length[src] = 0;
        p_num_shortest_paths[src] = 1;
        p_propagation_flag[src] = true;
      }
      p_num_predecessors[src] = 0;
      p_num_successors[src] = 0;
      p_dependency[src] = 0;
    }
  }
  // FP: "21 -> 22;
}
__global__ void FirstIterationSSSP(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_current_length)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_FirstIterationSSSP;
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
        index_type current_edge;
        current_edge = ns +_np_j;
        {
          index_type dst;
          int edge_weight;
          uint32_t new_dist;
          dst = graph.getAbsDestination(current_edge);
          edge_weight = 1;
#ifndef __USE_BFS__
          edge_weight += graph.getAbsWeight(current_edge);
#endif
          new_dist = edge_weight + p_current_length[src];
          atomicTestMin(&p_current_length[dst], new_dist);
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
          index_type current_edge;
          current_edge = _np_w_start +_np_ii;
          {
            index_type dst;
            int edge_weight;
            uint32_t new_dist;
            dst = graph.getAbsDestination(current_edge);
            edge_weight = 1;
#ifndef __USE_BFS__
            edge_weight += graph.getAbsWeight(current_edge);
#endif
            new_dist = edge_weight + p_current_length[src];
            atomicTestMin(&p_current_length[dst], new_dist);
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
        index_type current_edge;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        current_edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          int edge_weight;
          uint32_t new_dist;
          dst = graph.getAbsDestination(current_edge);
          edge_weight = 1;
#ifndef __USE_BFS__
          edge_weight += graph.getAbsWeight(current_edge);
#endif
          new_dist = edge_weight + p_current_length[src];
          atomicTestMin(&p_current_length[dst], new_dist);
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
__global__ void SSSP(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_current_length, uint32_t * p_old_length, HGAccumulator<uint32_t> DGAccumulator_accum)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_SSSP;
  __shared__ cub::BlockReduce<uint32_t, TB_SIZE>::TempStorage DGAccumulator_accum_ts;
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
  DGAccumulator_accum.thread_entry();
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
      if (p_old_length[src] > p_current_length[src])
      {
        p_old_length[src] = p_current_length[src];
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
        index_type current_edge;
        current_edge = ns +_np_j;
        {
          index_type dst;
          int edge_weight;
          uint32_t new_dist;
          uint32_t old;
          dst = graph.getAbsDestination(current_edge);
          edge_weight = 1;
#ifndef __USE_BFS__
          edge_weight += graph.getAbsWeight(current_edge);
#endif
          new_dist = edge_weight + p_current_length[src];
          old = atomicTestMin(&p_current_length[dst], new_dist);
          if (old > new_dist)
          {
            DGAccumulator_accum.reduce( 1);
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
        src = _np_closure[nps.warp.src[warpid]].src;
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type current_edge;
          current_edge = _np_w_start +_np_ii;
          {
            index_type dst;
            int edge_weight;
            uint32_t new_dist;
            uint32_t old;
            dst = graph.getAbsDestination(current_edge);
            edge_weight = 1;
#ifndef __USE_BFS__
            edge_weight += graph.getAbsWeight(current_edge);
#endif
            new_dist = edge_weight + p_current_length[src];
            old = atomicTestMin(&p_current_length[dst], new_dist);
            if (old > new_dist)
            {
              DGAccumulator_accum.reduce( 1);
            }
          }
        }
      }
      // FP: "97 -> 98;
      __syncthreads();
      // FP: "98 -> 99;
    }

    // FP: "99 -> 100;
    __syncthreads();
    // FP: "100 -> 101;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "101 -> 102;
    while (_np.work())
    {
      // FP: "102 -> 103;
      int _np_i =0;
      // FP: "103 -> 104;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "104 -> 105;
      __syncthreads();
      // FP: "105 -> 106;

      // FP: "106 -> 107;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type current_edge;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        current_edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          int edge_weight;
          uint32_t new_dist;
          uint32_t old;
          dst = graph.getAbsDestination(current_edge);
          edge_weight = 1;
#ifndef __USE_BFS__
          edge_weight += graph.getAbsWeight(current_edge);
#endif
          new_dist = edge_weight + p_current_length[src];
          old = atomicTestMin(&p_current_length[dst], new_dist);
          if (old > new_dist)
          {
            DGAccumulator_accum.reduce( 1);
          }
        }
      }
      // FP: "124 -> 125;
      _np.execute_round_done(ITSIZE);
      // FP: "125 -> 126;
      __syncthreads();
    }
    // FP: "127 -> 128;
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
  }
  // FP: "130 -> 131;
  DGAccumulator_accum.thread_exit<cub::BlockReduce<uint32_t, TB_SIZE>>(DGAccumulator_accum_ts);
  // FP: "131 -> 132;
}
__global__ void PredAndSucc(CSRGraph graph, unsigned int __begin, unsigned int __end, const uint32_t  local_infinity, uint32_t * p_current_length, uint32_t * p_num_predecessors, uint32_t * p_num_successors)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_PredAndSucc;
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
      if (p_current_length[src] != local_infinity)
      {
      }
      else
      {
        pop = false;
      }
    }
    // FP: "12 -> 13;
    // FP: "15 -> 16;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "16 -> 17;
    __shared__ struct { index_type src; } _np_closure [TB_SIZE];
    // FP: "17 -> 18;
    _np_closure[threadIdx.x].src = src;
    // FP: "18 -> 19;
    if (pop)
    {
      _np.size = (graph).getOutDegree(src);
      _np.start = (graph).getFirstEdge(src);
    }
    // FP: "21 -> 22;
    // FP: "22 -> 23;
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    // FP: "23 -> 24;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    // FP: "24 -> 25;
    if (threadIdx.x == 0)
    {
      nps.tb.owner = MAX_TB_SIZE + 1;
    }
    // FP: "27 -> 28;
    __syncthreads();
    // FP: "28 -> 29;
    while (true)
    {
      // FP: "29 -> 30;
      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }
      // FP: "32 -> 33;
      __syncthreads();
      // FP: "33 -> 34;
      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {
        // FP: "34 -> 35;
        __syncthreads();
        // FP: "35 -> 36;
        break;
      }
      // FP: "37 -> 38;
      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }
      // FP: "40 -> 41;
      __syncthreads();
      // FP: "41 -> 42;
      int ns = nps.tb.start;
      int ne = nps.tb.size;
      // FP: "42 -> 43;
      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }
      // FP: "45 -> 46;
      assert(nps.tb.src < __kernel_tb_size);
      src = _np_closure[nps.tb.src].src;
      // FP: "46 -> 47;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type current_edge;
        current_edge = ns +_np_j;
        {
          index_type dst;
          int edge_weight;
          dst = graph.getAbsDestination(current_edge);
          edge_weight = 1;
#ifndef __USE_BFS__
          edge_weight += graph.getAbsWeight(current_edge);
#endif
          if ((p_current_length[src] + edge_weight) == p_current_length[dst])
          {
            atomicTestAdd(&p_num_successors[src], (unsigned int)1);
            atomicTestAdd(&p_num_predecessors[dst], (unsigned int)1);
          }
        }
      }
      // FP: "61 -> 62;
      __syncthreads();
    }
    // FP: "63 -> 64;

    // FP: "64 -> 65;
    {
      const int warpid = threadIdx.x / 32;
      // FP: "65 -> 66;
      const int _np_laneid = cub::LaneId();
      // FP: "66 -> 67;
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
          index_type current_edge;
          current_edge = _np_w_start +_np_ii;
          {
            index_type dst;
            int edge_weight;
            dst = graph.getAbsDestination(current_edge);
            edge_weight = 1;
#ifndef __USE_BFS__
            edge_weight += graph.getAbsWeight(current_edge);
#endif
            if ((p_current_length[src] + edge_weight) == p_current_length[dst])
            {
              atomicTestAdd(&p_num_successors[src], (unsigned int)1);
              atomicTestAdd(&p_num_predecessors[dst], (unsigned int)1);
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
        index_type current_edge;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        current_edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          int edge_weight;
          dst = graph.getAbsDestination(current_edge);
          edge_weight = 1;
#ifndef __USE_BFS__
          edge_weight += graph.getAbsWeight(current_edge);
#endif
          if ((p_current_length[src] + edge_weight) == p_current_length[dst])
          {
            atomicTestAdd(&p_num_successors[src], (unsigned int)1);
            atomicTestAdd(&p_num_predecessors[dst], (unsigned int)1);
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
    src = _np_closure[threadIdx.x].src;
  }
  // FP: "121 -> 122;
}
__global__ void NumShortestPathsChanges(CSRGraph graph, unsigned int __begin, unsigned int __end, const uint32_t  local_infinity, uint32_t * p_current_length, uint32_t * p_num_predecessors, uint64_t * p_num_shortest_paths, uint8_t * p_propagation_flag, uint64_t * p_to_add, uint32_t * p_trim)
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
      if (p_current_length[src] != local_infinity)
      {
        if (p_trim[src] > 0)
        {
          p_num_predecessors[src] = p_num_predecessors[src] - p_trim[src];
          p_trim[src] = 0;
          if (p_num_predecessors[src] == 0)
          {
            p_propagation_flag[src] = true;
          }
        }
        if (p_to_add[src] > 0)
        {
          p_num_shortest_paths[src] += p_to_add[src];
          p_to_add[src] = 0;
        }
      }
    }
  }
  // FP: "20 -> 21;
}
__global__ void NumShortestPaths(CSRGraph graph, unsigned int __begin, unsigned int __end, const uint32_t  local_infinity, uint32_t * p_current_length, uint64_t * p_num_shortest_paths, uint8_t * p_propagation_flag, uint64_t * p_to_add, uint32_t * p_trim, HGAccumulator<uint32_t> DGAccumulator_accum)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_NumShortestPaths;
  __shared__ cub::BlockReduce<uint32_t, TB_SIZE>::TempStorage DGAccumulator_accum_ts;
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
  DGAccumulator_accum.thread_entry();
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
      if (p_current_length[src] != local_infinity)
      {
        if (p_propagation_flag[src])
        {
          p_propagation_flag[src] = false;
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
    __shared__ struct { index_type src; } _np_closure [TB_SIZE];
    // FP: "21 -> 22;
    _np_closure[threadIdx.x].src = src;
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
      src = _np_closure[nps.tb.src].src;
      // FP: "50 -> 51;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type current_edge;
        current_edge = ns +_np_j;
        {
          index_type dst;
          int edge_weight;
          uint64_t paths_to_add;
          dst = graph.getAbsDestination(current_edge);
          edge_weight = 1;
#ifndef __USE_BFS__
          edge_weight += graph.getAbsWeight(current_edge);
#endif
          paths_to_add = p_num_shortest_paths[src];
          if ((p_current_length[src] + edge_weight) == p_current_length[dst])
          {
            atomicTestAdd(&p_to_add[dst], paths_to_add);
            atomicTestAdd(&p_trim[dst], (unsigned int)1);
            DGAccumulator_accum.reduce( 1);
          }
        }
      }
      // FP: "68 -> 69;
      __syncthreads();
    }
    // FP: "70 -> 71;

    // FP: "71 -> 72;
    {
      const int warpid = threadIdx.x / 32;
      // FP: "72 -> 73;
      const int _np_laneid = cub::LaneId();
      // FP: "73 -> 74;
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
          index_type current_edge;
          current_edge = _np_w_start +_np_ii;
          {
            index_type dst;
            int edge_weight;
            uint64_t paths_to_add;
            dst = graph.getAbsDestination(current_edge);
            edge_weight = 1;
#ifndef __USE_BFS__
            edge_weight += graph.getAbsWeight(current_edge);
#endif
            paths_to_add = p_num_shortest_paths[src];
            if ((p_current_length[src] + edge_weight) == p_current_length[dst])
            {
              atomicTestAdd(&p_to_add[dst], paths_to_add);
              atomicTestAdd(&p_trim[dst], (unsigned int)1);
              DGAccumulator_accum.reduce( 1);
            }
          }
        }
      }
      // FP: "101 -> 102;
      __syncthreads();
      // FP: "102 -> 103;
    }

    // FP: "103 -> 104;
    __syncthreads();
    // FP: "104 -> 105;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "105 -> 106;
    while (_np.work())
    {
      // FP: "106 -> 107;
      int _np_i =0;
      // FP: "107 -> 108;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "108 -> 109;
      __syncthreads();
      // FP: "109 -> 110;

      // FP: "110 -> 111;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type current_edge;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        current_edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          int edge_weight;
          uint64_t paths_to_add;
          dst = graph.getAbsDestination(current_edge);
          edge_weight = 1;
#ifndef __USE_BFS__
          edge_weight += graph.getAbsWeight(current_edge);
#endif
          paths_to_add = p_num_shortest_paths[src];
          if ((p_current_length[src] + edge_weight) == p_current_length[dst])
          {
            atomicTestAdd(&p_to_add[dst], paths_to_add);
            atomicTestAdd(&p_trim[dst], (unsigned int)1);
            DGAccumulator_accum.reduce( 1);
          }
        }
      }
      // FP: "129 -> 130;
      _np.execute_round_done(ITSIZE);
      // FP: "130 -> 131;
      __syncthreads();
    }
    // FP: "132 -> 133;
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
  }
  // FP: "136 -> 137;
  DGAccumulator_accum.thread_exit<cub::BlockReduce<uint32_t, TB_SIZE>>(DGAccumulator_accum_ts);
  // FP: "137 -> 138;
}
__global__ void PropagationFlagUpdate(CSRGraph graph, unsigned int __begin, unsigned int __end, const uint32_t  local_infinity, uint32_t * p_current_length, uint32_t * p_num_successors, uint8_t * p_propagation_flag)
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
      if (p_current_length[src] != local_infinity)
      {
        if (p_num_successors[src] == 0)
        {
          p_propagation_flag[src] = true;
        }
      }
    }
  }
  // FP: "12 -> 13;
}
__global__ void DependencyPropChanges(CSRGraph graph, unsigned int __begin, unsigned int __end, const uint32_t  local_infinity, uint32_t * p_current_length, float * p_dependency, uint32_t * p_num_successors, uint8_t * p_propagation_flag, float * p_to_add_float, uint32_t * p_trim)
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
      if (p_current_length[src] != local_infinity)
      {
        if (p_to_add_float[src] > 0.0)
        {
          p_dependency[src] += p_to_add_float[src];
          p_to_add_float[src] = 0.0;
        }
        if (p_num_successors[src] == 0 && p_propagation_flag[src])
        {
          p_propagation_flag[src] = false;
        }
        else
        {
          if (p_trim[src] > 0)
          {
            p_num_successors[src] = p_num_successors[src] - p_trim[src];
            p_trim[src] = 0;
            if (p_num_successors[src] == 0)
            {
              p_propagation_flag[src] = true;
            }
          }
        }
      }
    }
  }
  // FP: "25 -> 26;
}
__global__ void DependencyPropagation(CSRGraph graph, unsigned int __begin, unsigned int __end, const uint64_t  local_current_src_node, const uint32_t  local_infinity, uint32_t * p_current_length, float * p_dependency, uint64_t * p_num_shortest_paths, uint32_t * p_num_successors, uint8_t * p_propagation_flag, float * p_to_add_float, uint32_t * p_trim, HGAccumulator<uint32_t> DGAccumulator_accum)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_DependencyPropagation;
  __shared__ cub::BlockReduce<uint32_t, TB_SIZE>::TempStorage DGAccumulator_accum_ts;
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
  DGAccumulator_accum.thread_entry();
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
      if (p_current_length[src] != local_infinity)
      {
        if (p_num_successors[src] > 0)
        {
          if (graph.node_data[src] == local_current_src_node)
          {
            p_num_successors[src] = 0;
          }
          if (graph.node_data[src] != local_current_src_node)
          {
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
      else
      {
        pop = false;
      }
    }
    // FP: "21 -> 22;
    // FP: "24 -> 25;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "25 -> 26;
    __shared__ struct { index_type src; } _np_closure [TB_SIZE];
    // FP: "26 -> 27;
    _np_closure[threadIdx.x].src = src;
    // FP: "27 -> 28;
    if (pop)
    {
      _np.size = (graph).getOutDegree(src);
      _np.start = (graph).getFirstEdge(src);
    }
    // FP: "30 -> 31;
    // FP: "31 -> 32;
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    // FP: "32 -> 33;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    // FP: "33 -> 34;
    if (threadIdx.x == 0)
    {
      nps.tb.owner = MAX_TB_SIZE + 1;
    }
    // FP: "36 -> 37;
    __syncthreads();
    // FP: "37 -> 38;
    while (true)
    {
      // FP: "38 -> 39;
      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }
      // FP: "41 -> 42;
      __syncthreads();
      // FP: "42 -> 43;
      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {
        // FP: "43 -> 44;
        __syncthreads();
        // FP: "44 -> 45;
        break;
      }
      // FP: "46 -> 47;
      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }
      // FP: "49 -> 50;
      __syncthreads();
      // FP: "50 -> 51;
      int ns = nps.tb.start;
      int ne = nps.tb.size;
      // FP: "51 -> 52;
      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }
      // FP: "54 -> 55;
      assert(nps.tb.src < __kernel_tb_size);
      src = _np_closure[nps.tb.src].src;
      // FP: "55 -> 56;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type current_edge;
        current_edge = ns +_np_j;
        {
          index_type dst;
          int edge_weight;
          dst = graph.getAbsDestination(current_edge);
          edge_weight = 1;
#ifndef __USE_BFS__
          edge_weight += graph.getAbsWeight(current_edge);
#endif
          if (p_propagation_flag[dst])
          {
            if ((p_current_length[src] + edge_weight) == p_current_length[dst])
            {
              float contrib;
              atomicTestAdd(&p_trim[src], (unsigned int)1);
              contrib = p_num_shortest_paths[src];
              contrib /= p_num_shortest_paths[dst];
              contrib *= (1.0 + p_dependency[dst]);
              atomicTestAdd(&p_to_add_float[src], contrib);
              DGAccumulator_accum.reduce( 1);
            }
          }
        }
      }
      // FP: "77 -> 78;
      __syncthreads();
    }
    // FP: "79 -> 80;

    // FP: "80 -> 81;
    {
      const int warpid = threadIdx.x / 32;
      // FP: "81 -> 82;
      const int _np_laneid = cub::LaneId();
      // FP: "82 -> 83;
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
          index_type current_edge;
          current_edge = _np_w_start +_np_ii;
          {
            index_type dst;
            int edge_weight;
            dst = graph.getAbsDestination(current_edge);
            edge_weight = 1;
#ifndef __USE_BFS__
            edge_weight += graph.getAbsWeight(current_edge);
#endif
            if (p_propagation_flag[dst])
            {
              if ((p_current_length[src] + edge_weight) == p_current_length[dst])
              {
                float contrib;
                atomicTestAdd(&p_trim[src], (unsigned int)1);
                contrib = p_num_shortest_paths[src];
                contrib /= p_num_shortest_paths[dst];
                contrib *= (1.0 + p_dependency[dst]);
                atomicTestAdd(&p_to_add_float[src], contrib);
                DGAccumulator_accum.reduce( 1);
              }
            }
          }
        }
      }
      // FP: "114 -> 115;
      __syncthreads();
      // FP: "115 -> 116;
    }

    // FP: "116 -> 117;
    __syncthreads();
    // FP: "117 -> 118;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "118 -> 119;
    while (_np.work())
    {
      // FP: "119 -> 120;
      int _np_i =0;
      // FP: "120 -> 121;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "121 -> 122;
      __syncthreads();
      // FP: "122 -> 123;

      // FP: "123 -> 124;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type current_edge;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        current_edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          int edge_weight;
          dst = graph.getAbsDestination(current_edge);
          edge_weight = 1;
#ifndef __USE_BFS__
          edge_weight += graph.getAbsWeight(current_edge);
#endif
          if (p_propagation_flag[dst])
          {
            if ((p_current_length[src] + edge_weight) == p_current_length[dst])
            {
              float contrib;
              atomicTestAdd(&p_trim[src], (unsigned int)1);
              contrib = p_num_shortest_paths[src];
              contrib /= p_num_shortest_paths[dst];
              contrib *= (1.0 + p_dependency[dst]);
              atomicTestAdd(&p_to_add_float[src], contrib);
              DGAccumulator_accum.reduce( 1);
            }
          }
        }
      }
      // FP: "146 -> 147;
      _np.execute_round_done(ITSIZE);
      // FP: "147 -> 148;
      __syncthreads();
    }
    // FP: "149 -> 150;
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
  }
  // FP: "153 -> 154;
  DGAccumulator_accum.thread_exit<cub::BlockReduce<uint32_t, TB_SIZE>>(DGAccumulator_accum_ts);
  // FP: "154 -> 155;
}
__global__ void BC(CSRGraph graph, unsigned int __begin, unsigned int __end, float * p_betweeness_centrality, float * p_dependency)
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
      if (p_dependency[src] > 0)
      {
        atomicTestAdd(&p_betweeness_centrality[src], p_dependency[src]);
      }
    }
  }
  // FP: "9 -> 10;
}
__global__ void Sanity(CSRGraph graph, unsigned int __begin, unsigned int __end, float * p_betweeness_centrality, HGAccumulator<float> DGAccumulator_sum, HGReduceMax<float> DGAccumulator_max, HGReduceMin<float> DGAccumulator_min)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  __shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage DGAccumulator_sum_ts;
  __shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage DGAccumulator_max_ts;
  __shared__ cub::BlockReduce<float, TB_SIZE>::TempStorage DGAccumulator_min_ts;
  index_type src_end;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  DGAccumulator_sum.thread_entry();
  // FP: "3 -> 4;
  // FP: "4 -> 5;
  DGAccumulator_max.thread_entry();
  // FP: "5 -> 6;
  // FP: "6 -> 7;
  DGAccumulator_min.thread_entry();
  // FP: "7 -> 8;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      DGAccumulator_max.reduce(p_betweeness_centrality[src]);
      DGAccumulator_min.reduce(p_betweeness_centrality[src]);
      DGAccumulator_sum.reduce( p_betweeness_centrality[src]);
    }
  }
  // FP: "15 -> 16;
  DGAccumulator_sum.thread_exit<cub::BlockReduce<float, TB_SIZE>>(DGAccumulator_sum_ts);
  // FP: "16 -> 17;
  DGAccumulator_max.thread_exit<cub::BlockReduce<float, TB_SIZE>>(DGAccumulator_max_ts);
  // FP: "17 -> 18;
  DGAccumulator_min.thread_exit<cub::BlockReduce<float, TB_SIZE>>(DGAccumulator_min_ts);
  // FP: "18 -> 19;
}
void InitializeGraph_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  InitializeGraph <<<blocks, threads>>>(ctx->gg, __begin, __end, ctx->betweeness_centrality.data.gpu_wr_ptr(), ctx->dependency.data.gpu_wr_ptr(), ctx->num_predecessors.data.gpu_wr_ptr(), ctx->num_shortest_paths.data.gpu_wr_ptr(), ctx->num_successors.data.gpu_wr_ptr(), ctx->propagation_flag.data.gpu_wr_ptr(), ctx->to_add.data.gpu_wr_ptr(), ctx->to_add_float.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr());
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
void InitializeIteration_cuda(unsigned int  __begin, unsigned int  __end, const uint32_t & local_infinity, const uint64_t & local_current_src_node, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  InitializeIteration <<<blocks, threads>>>(ctx->gg, __begin, __end, local_current_src_node, local_infinity, ctx->current_length.data.gpu_wr_ptr(), ctx->dependency.data.gpu_wr_ptr(), ctx->num_predecessors.data.gpu_wr_ptr(), ctx->num_shortest_paths.data.gpu_wr_ptr(), ctx->num_successors.data.gpu_wr_ptr(), ctx->old_length.data.gpu_wr_ptr(), ctx->propagation_flag.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void InitializeIteration_allNodes_cuda(const uint32_t & local_infinity, const uint64_t & local_current_src_node, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  InitializeIteration_cuda(0, ctx->gg.nnodes, local_infinity, local_current_src_node, ctx);
  // FP: "2 -> 3;
}
void InitializeIteration_masterNodes_cuda(const uint32_t & local_infinity, const uint64_t & local_current_src_node, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  InitializeIteration_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_infinity, local_current_src_node, ctx);
  // FP: "2 -> 3;
}
void InitializeIteration_nodesWithEdges_cuda(const uint32_t & local_infinity, const uint64_t & local_current_src_node, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  InitializeIteration_cuda(0, ctx->numNodesWithEdges, local_infinity, local_current_src_node, ctx);
  // FP: "2 -> 3;
}
void FirstIterationSSSP_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  FirstIterationSSSP <<<blocks, __tb_FirstIterationSSSP>>>(ctx->gg, __begin, __end, ctx->current_length.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void FirstIterationSSSP_allNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  FirstIterationSSSP_cuda(0, ctx->gg.nnodes, ctx);
  // FP: "2 -> 3;
}
void FirstIterationSSSP_masterNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  FirstIterationSSSP_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx);
  // FP: "2 -> 3;
}
void FirstIterationSSSP_nodesWithEdges_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  FirstIterationSSSP_cuda(0, ctx->numNodesWithEdges, ctx);
  // FP: "2 -> 3;
}
void SSSP_cuda(unsigned int  __begin, unsigned int  __end, uint32_t & DGAccumulator_accum, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  HGAccumulator<uint32_t> _DGAccumulator_accum;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  Shared<uint32_t> DGAccumulator_accumval  = Shared<uint32_t>(1);
  // FP: "5 -> 6;
  // FP: "6 -> 7;
  *(DGAccumulator_accumval.cpu_wr_ptr()) = 0;
  // FP: "7 -> 8;
  _DGAccumulator_accum.rv = DGAccumulator_accumval.gpu_wr_ptr();
  // FP: "8 -> 9;
  SSSP <<<blocks, __tb_SSSP>>>(ctx->gg, __begin, __end, ctx->current_length.data.gpu_wr_ptr(), ctx->old_length.data.gpu_wr_ptr(), _DGAccumulator_accum);
  // FP: "9 -> 10;
  check_cuda_kernel;
  // FP: "10 -> 11;
  DGAccumulator_accum = *(DGAccumulator_accumval.cpu_rd_ptr());
  // FP: "11 -> 12;
}
void SSSP_allNodes_cuda(uint32_t & DGAccumulator_accum, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  SSSP_cuda(0, ctx->gg.nnodes, DGAccumulator_accum, ctx);
  // FP: "2 -> 3;
}
void SSSP_masterNodes_cuda(uint32_t & DGAccumulator_accum, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  SSSP_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, DGAccumulator_accum, ctx);
  // FP: "2 -> 3;
}
void SSSP_nodesWithEdges_cuda(uint32_t & DGAccumulator_accum, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  SSSP_cuda(0, ctx->numNodesWithEdges, DGAccumulator_accum, ctx);
  // FP: "2 -> 3;
}
void PredAndSucc_cuda(unsigned int  __begin, unsigned int  __end, const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  PredAndSucc <<<blocks, __tb_PredAndSucc>>>(ctx->gg, __begin, __end, local_infinity, ctx->current_length.data.gpu_wr_ptr(), ctx->num_predecessors.data.gpu_wr_ptr(), ctx->num_successors.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void PredAndSucc_allNodes_cuda(const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  PredAndSucc_cuda(0, ctx->gg.nnodes, local_infinity, ctx);
  // FP: "2 -> 3;
}
void PredAndSucc_masterNodes_cuda(const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  PredAndSucc_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_infinity, ctx);
  // FP: "2 -> 3;
}
void PredAndSucc_nodesWithEdges_cuda(const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  PredAndSucc_cuda(0, ctx->numNodesWithEdges, local_infinity, ctx);
  // FP: "2 -> 3;
}
void NumShortestPathsChanges_cuda(unsigned int  __begin, unsigned int  __end, const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  NumShortestPathsChanges <<<blocks, threads>>>(ctx->gg, __begin, __end, local_infinity, ctx->current_length.data.gpu_wr_ptr(), ctx->num_predecessors.data.gpu_wr_ptr(), ctx->num_shortest_paths.data.gpu_wr_ptr(), ctx->propagation_flag.data.gpu_wr_ptr(), ctx->to_add.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void NumShortestPathsChanges_allNodes_cuda(const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  NumShortestPathsChanges_cuda(0, ctx->gg.nnodes, local_infinity, ctx);
  // FP: "2 -> 3;
}
void NumShortestPathsChanges_masterNodes_cuda(const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  NumShortestPathsChanges_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_infinity, ctx);
  // FP: "2 -> 3;
}
void NumShortestPathsChanges_nodesWithEdges_cuda(const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  NumShortestPathsChanges_cuda(0, ctx->numNodesWithEdges, local_infinity, ctx);
  // FP: "2 -> 3;
}
void NumShortestPaths_cuda(unsigned int  __begin, unsigned int  __end, uint32_t & DGAccumulator_accum, const uint32_t & local_infinity, const uint64_t local_current_src_node, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  HGAccumulator<uint32_t> _DGAccumulator_accum;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  Shared<uint32_t> DGAccumulator_accumval  = Shared<uint32_t>(1);
  // FP: "5 -> 6;
  // FP: "6 -> 7;
  *(DGAccumulator_accumval.cpu_wr_ptr()) = 0;
  // FP: "7 -> 8;
  _DGAccumulator_accum.rv = DGAccumulator_accumval.gpu_wr_ptr();
  // FP: "8 -> 9;
  NumShortestPaths <<<blocks, __tb_NumShortestPaths>>>(ctx->gg, __begin, __end, local_infinity, ctx->current_length.data.gpu_wr_ptr(), ctx->num_shortest_paths.data.gpu_wr_ptr(), ctx->propagation_flag.data.gpu_wr_ptr(), ctx->to_add.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr(), _DGAccumulator_accum);
  // FP: "9 -> 10;
  check_cuda_kernel;
  // FP: "10 -> 11;
  DGAccumulator_accum = *(DGAccumulator_accumval.cpu_rd_ptr());
  // FP: "11 -> 12;
}
void NumShortestPaths_allNodes_cuda(uint32_t & DGAccumulator_accum, const uint32_t & local_infinity, const uint64_t local_current_src_node, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  NumShortestPaths_cuda(0, ctx->gg.nnodes, DGAccumulator_accum, local_infinity, local_current_src_node, ctx);
  // FP: "2 -> 3;
}
void NumShortestPaths_masterNodes_cuda(uint32_t & DGAccumulator_accum, const uint32_t & local_infinity, const uint64_t local_current_src_node, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  NumShortestPaths_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, DGAccumulator_accum, local_infinity, local_current_src_node, ctx);
  // FP: "2 -> 3;
}
void NumShortestPaths_nodesWithEdges_cuda(uint32_t & DGAccumulator_accum, const uint32_t & local_infinity, const uint64_t local_current_src_node, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  NumShortestPaths_cuda(0, ctx->numNodesWithEdges, DGAccumulator_accum, local_infinity, local_current_src_node, ctx);
  // FP: "2 -> 3;
}
void PropagationFlagUpdate_cuda(unsigned int  __begin, unsigned int  __end, const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  PropagationFlagUpdate <<<blocks, threads>>>(ctx->gg, __begin, __end, local_infinity, ctx->current_length.data.gpu_wr_ptr(), ctx->num_successors.data.gpu_wr_ptr(), ctx->propagation_flag.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void PropagationFlagUpdate_allNodes_cuda(const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  PropagationFlagUpdate_cuda(0, ctx->gg.nnodes, local_infinity, ctx);
  // FP: "2 -> 3;
}
void PropagationFlagUpdate_masterNodes_cuda(const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  PropagationFlagUpdate_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_infinity, ctx);
  // FP: "2 -> 3;
}
void PropagationFlagUpdate_nodesWithEdges_cuda(const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  PropagationFlagUpdate_cuda(0, ctx->numNodesWithEdges, local_infinity, ctx);
  // FP: "2 -> 3;
}
void DependencyPropChanges_cuda(unsigned int  __begin, unsigned int  __end, const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  DependencyPropChanges <<<blocks, threads>>>(ctx->gg, __begin, __end, local_infinity, ctx->current_length.data.gpu_wr_ptr(), ctx->dependency.data.gpu_wr_ptr(), ctx->num_successors.data.gpu_wr_ptr(), ctx->propagation_flag.data.gpu_wr_ptr(), ctx->to_add_float.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void DependencyPropChanges_allNodes_cuda(const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  DependencyPropChanges_cuda(0, ctx->gg.nnodes, local_infinity, ctx);
  // FP: "2 -> 3;
}
void DependencyPropChanges_masterNodes_cuda(const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  DependencyPropChanges_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_infinity, ctx);
  // FP: "2 -> 3;
}
void DependencyPropChanges_nodesWithEdges_cuda(const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  DependencyPropChanges_cuda(0, ctx->numNodesWithEdges, local_infinity, ctx);
  // FP: "2 -> 3;
}
void DependencyPropagation_cuda(unsigned int  __begin, unsigned int  __end, uint32_t & DGAccumulator_accum, const uint32_t & local_infinity, const uint64_t & local_current_src_node, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  HGAccumulator<uint32_t> _DGAccumulator_accum;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  Shared<uint32_t> DGAccumulator_accumval  = Shared<uint32_t>(1);
  // FP: "5 -> 6;
  // FP: "6 -> 7;
  *(DGAccumulator_accumval.cpu_wr_ptr()) = 0;
  // FP: "7 -> 8;
  _DGAccumulator_accum.rv = DGAccumulator_accumval.gpu_wr_ptr();
  // FP: "8 -> 9;
  DependencyPropagation <<<blocks, __tb_DependencyPropagation>>>(ctx->gg, __begin, __end, local_current_src_node, local_infinity, ctx->current_length.data.gpu_wr_ptr(), ctx->dependency.data.gpu_wr_ptr(), ctx->num_shortest_paths.data.gpu_wr_ptr(), ctx->num_successors.data.gpu_wr_ptr(), ctx->propagation_flag.data.gpu_wr_ptr(), ctx->to_add_float.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr(), _DGAccumulator_accum);
  // FP: "9 -> 10;
  check_cuda_kernel;
  // FP: "10 -> 11;
  DGAccumulator_accum = *(DGAccumulator_accumval.cpu_rd_ptr());
  // FP: "11 -> 12;
}
void DependencyPropagation_allNodes_cuda(uint32_t & DGAccumulator_accum, const uint32_t & local_infinity, const uint64_t & local_current_src_node, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  DependencyPropagation_cuda(0, ctx->gg.nnodes, DGAccumulator_accum, local_infinity, local_current_src_node, ctx);
  // FP: "2 -> 3;
}
void DependencyPropagation_masterNodes_cuda(uint32_t & DGAccumulator_accum, const uint32_t & local_infinity, const uint64_t & local_current_src_node, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  DependencyPropagation_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, DGAccumulator_accum, local_infinity, local_current_src_node, ctx);
  // FP: "2 -> 3;
}
void DependencyPropagation_nodesWithEdges_cuda(uint32_t & DGAccumulator_accum, const uint32_t & local_infinity, const uint64_t & local_current_src_node, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  DependencyPropagation_cuda(0, ctx->numNodesWithEdges, DGAccumulator_accum, local_infinity, local_current_src_node, ctx);
  // FP: "2 -> 3;
}
void BC_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  BC <<<blocks, threads>>>(ctx->gg, __begin, __end, ctx->betweeness_centrality.data.gpu_wr_ptr(), ctx->dependency.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void BC_allNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  BC_cuda(0, ctx->gg.nnodes, ctx);
  // FP: "2 -> 3;
}
void BC_masterNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  BC_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx);
  // FP: "2 -> 3;
}
void BC_nodesWithEdges_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  BC_cuda(0, ctx->numNodesWithEdges, ctx);
  // FP: "2 -> 3;
}
void Sanity_cuda(unsigned int  __begin, unsigned int  __end, float & DGAccumulator_sum, float & DGAccumulator_max, float & DGAccumulator_min, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  HGAccumulator<float> _DGAccumulator_sum;
  HGReduceMax<float> _DGAccumulator_max;
  HGReduceMin<float> _DGAccumulator_min;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  Shared<float> DGAccumulator_sumval  = Shared<float>(1);
  // FP: "5 -> 6;
  // FP: "6 -> 7;
  *(DGAccumulator_sumval.cpu_wr_ptr()) = 0;
  // FP: "7 -> 8;
  _DGAccumulator_sum.rv = DGAccumulator_sumval.gpu_wr_ptr();
  // FP: "8 -> 9;
  Shared<float> DGAccumulator_maxval  = Shared<float>(1);
  // FP: "9 -> 10;
  // FP: "10 -> 11;
  *(DGAccumulator_maxval.cpu_wr_ptr()) = 0;
  // FP: "11 -> 12;
  _DGAccumulator_max.rv = DGAccumulator_maxval.gpu_wr_ptr();
  // FP: "12 -> 13;
  Shared<float> DGAccumulator_minval  = Shared<float>(1);
  // FP: "13 -> 14;
  // FP: "14 -> 15;
  *(DGAccumulator_minval.cpu_wr_ptr()) = 0;
  // FP: "15 -> 16;
  _DGAccumulator_min.rv = DGAccumulator_minval.gpu_wr_ptr();
  // FP: "16 -> 17;
  Sanity <<<blocks, threads>>>(ctx->gg, __begin, __end, ctx->betweeness_centrality.data.gpu_wr_ptr(), _DGAccumulator_sum, _DGAccumulator_max, _DGAccumulator_min);
  // FP: "17 -> 18;
  check_cuda_kernel;
  // FP: "18 -> 19;
  DGAccumulator_sum = *(DGAccumulator_sumval.cpu_rd_ptr());
  // FP: "19 -> 20;
  DGAccumulator_max = *(DGAccumulator_maxval.cpu_rd_ptr());
  // FP: "20 -> 21;
  DGAccumulator_min = *(DGAccumulator_minval.cpu_rd_ptr());
  // FP: "21 -> 22;
}
void Sanity_allNodes_cuda(float & DGAccumulator_sum, float & DGAccumulator_max, float & DGAccumulator_min, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  Sanity_cuda(0, ctx->gg.nnodes, DGAccumulator_sum, DGAccumulator_max, DGAccumulator_min, ctx);
  // FP: "2 -> 3;
}
void Sanity_masterNodes_cuda(float & DGAccumulator_sum, float & DGAccumulator_max, float & DGAccumulator_min, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  Sanity_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, DGAccumulator_sum, DGAccumulator_max, DGAccumulator_min, ctx);
  // FP: "2 -> 3;
}
void Sanity_nodesWithEdges_cuda(float & DGAccumulator_sum, float & DGAccumulator_max, float & DGAccumulator_min, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  Sanity_cuda(0, ctx->numNodesWithEdges, DGAccumulator_sum, DGAccumulator_max, DGAccumulator_min, ctx);
  // FP: "2 -> 3;
}
