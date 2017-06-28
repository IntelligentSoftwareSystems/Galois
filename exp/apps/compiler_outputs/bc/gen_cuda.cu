/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
float * P_BETWEENESS_CENTRALITY;
unsigned int * P_CURRENT_LENGTH;
float * P_DEPENDENCY;
unsigned int * P_NUM_PREDECESSORS;
unsigned int * P_NUM_SHORTEST_PATHS;
unsigned int * P_NUM_SUCCESSORS;
unsigned int * P_OLD_LENGTH;
bool * P_PROPOGATION_FLAG;
unsigned int * P_TRIM;
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
static const int __tb_NumShortestPaths = TB_SIZE;
static const int __tb_FirstIterationSSSP = TB_SIZE;
static const int __tb_SSSP = TB_SIZE;
static const int __tb_DependencyPropogation = TB_SIZE;
static const int __tb_PredAndSucc = TB_SIZE;
__global__ void InitializeGraph(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, float * p_betweeness_centrality)
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
    }
  }
  // FP: "7 -> 8;
}
__global__ void InitializeIteration(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, const unsigned int  local_current_src_node, const unsigned int  local_infinity, unsigned int * p_current_length, float * p_dependency, unsigned int * p_num_predecessors, unsigned int * p_num_shortest_paths, unsigned int * p_num_successors, unsigned int * p_old_length, bool * p_propogation_flag, unsigned int * p_trim)
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
      p_current_length[src] = (graph.node_data[src] == local_current_src_node) ?  0 : local_infinity;
      p_old_length[src] = (graph.node_data[src] == local_current_src_node) ?  0 : local_infinity;
      p_trim[src] = 0;
      p_propogation_flag[src] = false;
      p_num_shortest_paths[src] = (graph.node_data[src] == local_current_src_node) ?  1 : 0;
      p_num_successors[src] = 0;
      p_num_predecessors[src] = 0;
      p_dependency[src] = 0;
    }
  }
  // FP: "14 -> 15;
}
__global__ void FirstIterationSSSP(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, unsigned int * p_current_length)
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
          unsigned int new_dist;
          dst = graph.getAbsDestination(current_edge);
          new_dist = 1 + p_current_length[src];
          atomicMin(&p_current_length[dst], new_dist);
        }
      }
      // FP: "52 -> 53;
      __syncthreads();
    }
    // FP: "54 -> 55;

    // FP: "55 -> 56;
    {
      const int warpid = threadIdx.x / 32;
      // FP: "56 -> 57;
      const int _np_laneid = cub::LaneId();
      // FP: "57 -> 58;
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
            unsigned int new_dist;
            dst = graph.getAbsDestination(current_edge);
            new_dist = 1 + p_current_length[src];
            atomicMin(&p_current_length[dst], new_dist);
          }
        }
      }
      // FP: "76 -> 77;
      __syncthreads();
      // FP: "77 -> 78;
    }

    // FP: "78 -> 79;
    __syncthreads();
    // FP: "79 -> 80;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "80 -> 81;
    while (_np.work())
    {
      // FP: "81 -> 82;
      int _np_i =0;
      // FP: "82 -> 83;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "83 -> 84;
      __syncthreads();
      // FP: "84 -> 85;

      // FP: "85 -> 86;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type current_edge;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        current_edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          unsigned int new_dist;
          dst = graph.getAbsDestination(current_edge);
          new_dist = 1 + p_current_length[src];
          atomicMin(&p_current_length[dst], new_dist);
        }
      }
      // FP: "95 -> 96;
      _np.execute_round_done(ITSIZE);
      // FP: "96 -> 97;
      __syncthreads();
    }
    // FP: "98 -> 99;
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
  }
  // FP: "100 -> 101;
}
__global__ void SSSP(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, unsigned int * p_current_length, unsigned int * p_old_length, Sum ret_val)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_SSSP;
  typedef cub::BlockReduce<int, TB_SIZE> _br;
  __shared__ _br::TempStorage _ts;
  ret_val.thread_entry();
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
      if (p_old_length[src] > p_current_length[src])
      {
        p_old_length[src] = p_current_length[src];
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
          unsigned int new_dist;
          dst = graph.getAbsDestination(current_edge);
          new_dist = 1 + p_current_length[src];
          atomicMin(&p_current_length[dst], new_dist);
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
          index_type current_edge;
          current_edge = _np_w_start +_np_ii;
          {
            index_type dst;
            unsigned int new_dist;
            dst = graph.getAbsDestination(current_edge);
            new_dist = 1 + p_current_length[src];
            atomicMin(&p_current_length[dst], new_dist);
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
        index_type current_edge;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        current_edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          unsigned int new_dist;
          dst = graph.getAbsDestination(current_edge);
          new_dist = 1 + p_current_length[src];
          atomicMin(&p_current_length[dst], new_dist);
        }
      }
      // FP: "98 -> 99;
      _np.execute_round_done(ITSIZE);
      // FP: "99 -> 100;
      __syncthreads();
    }
    // FP: "101 -> 102;
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
    // FP: "102 -> 103;
    ret_val.do_return( 1);
    // FP: "103 -> 104;
    continue;
  }
  ret_val.thread_exit<_br>(_ts);
}
__global__ void PredAndSucc(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, unsigned int * p_current_length, unsigned int * p_num_predecessors, unsigned int * p_num_successors)
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
          unsigned int edge_weight;
          dst = graph.getAbsDestination(current_edge);
          edge_weight = 1;
          if ((p_current_length[src] + edge_weight) == p_current_length[dst])
          {
            atomicAdd(&p_num_successors[src], (unsigned int)1);
            atomicAdd(&p_num_predecessors[dst], (unsigned int)1);
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
          index_type current_edge;
          current_edge = _np_w_start +_np_ii;
          {
            index_type dst;
            unsigned int edge_weight;
            dst = graph.getAbsDestination(current_edge);
            edge_weight = 1;
            if ((p_current_length[src] + edge_weight) == p_current_length[dst])
            {
              atomicAdd(&p_num_successors[src], (unsigned int)1);
              atomicAdd(&p_num_predecessors[dst], (unsigned int)1);
            }
          }
        }
      }
      // FP: "82 -> 83;
      __syncthreads();
      // FP: "83 -> 84;
    }

    // FP: "84 -> 85;
    __syncthreads();
    // FP: "85 -> 86;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "86 -> 87;
    while (_np.work())
    {
      // FP: "87 -> 88;
      int _np_i =0;
      // FP: "88 -> 89;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "89 -> 90;
      __syncthreads();
      // FP: "90 -> 91;

      // FP: "91 -> 92;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type current_edge;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        current_edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          unsigned int edge_weight;
          dst = graph.getAbsDestination(current_edge);
          edge_weight = 1;
          if ((p_current_length[src] + edge_weight) == p_current_length[dst])
          {
            atomicAdd(&p_num_successors[src], (unsigned int)1);
            atomicAdd(&p_num_predecessors[dst], (unsigned int)1);
          }
        }
      }
      // FP: "104 -> 105;
      _np.execute_round_done(ITSIZE);
      // FP: "105 -> 106;
      __syncthreads();
    }
    // FP: "107 -> 108;
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
  }
  // FP: "109 -> 110;
}
__global__ void PredecessorDecrement(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, unsigned int * p_num_predecessors, bool * p_propogation_flag, unsigned int * p_trim)
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
      if (p_trim[src] > 0)
      {
        if (p_trim[src] > p_num_predecessors[src])
        {
          // DANGER
          //abort();
        }
        p_num_predecessors[src] -= p_trim[src];
        p_trim[src] = 0;
        if (p_num_predecessors[src] == 0)
        {
          p_propogation_flag[src] = false;
        }
      }
    }
  }
  // FP: "16 -> 17;
}
__global__ void NumShortestPaths(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, unsigned int * p_current_length, unsigned int * p_num_predecessors, unsigned int * p_num_shortest_paths, bool * p_propogation_flag, unsigned int * p_trim, Sum ret_val)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_NumShortestPaths;
  typedef cub::BlockReduce<int, TB_SIZE> _br;
  __shared__ _br::TempStorage _ts;
  ret_val.thread_entry();
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
    bool pop  = src < __end;
    if (pop)
    {
      if (p_num_predecessors[src] == 0 && !p_propogation_flag[src])
      {
      }
      else
      {
        pop = false;
      }
    }
    struct NPInspector1 _np = {0,0,0,0,0,0};
    __shared__ struct { index_type src; } _np_closure [TB_SIZE];
    _np_closure[threadIdx.x].src = src;
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
      src = _np_closure[nps.tb.src].src;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type current_edge;
        current_edge = ns +_np_j;
        {
          index_type dst;
          unsigned int edge_weight;
          unsigned int to_add;
          dst = graph.getAbsDestination(current_edge);
          edge_weight = 1;
          to_add = p_num_shortest_paths[src];
          if ((p_current_length[src] + edge_weight) == p_current_length[dst])
          {
            atomicAdd(&p_num_shortest_paths[dst], to_add);
            atomicAdd(&p_trim[dst], (unsigned int)1);
            ret_val.do_return( 1);
            continue;
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
        src = _np_closure[nps.warp.src[warpid]].src;
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type current_edge;
          current_edge = _np_w_start +_np_ii;
          {
            index_type dst;
            unsigned int edge_weight;
            unsigned int to_add;
            dst = graph.getAbsDestination(current_edge);
            edge_weight = 1;
            to_add = p_num_shortest_paths[src];
            if ((p_current_length[src] + edge_weight) == p_current_length[dst])
            {
              atomicAdd(&p_num_shortest_paths[dst], to_add);
              atomicAdd(&p_trim[dst], (unsigned int)1);
              ret_val.do_return( 1);
              continue;
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
        index_type current_edge;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        current_edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          unsigned int edge_weight;
          unsigned int to_add;
          dst = graph.getAbsDestination(current_edge);
          edge_weight = 1;
          to_add = p_num_shortest_paths[src];
          if ((p_current_length[src] + edge_weight) == p_current_length[dst])
          {
            atomicAdd(&p_num_shortest_paths[dst], to_add);
            atomicAdd(&p_trim[dst], (unsigned int)1);
            ret_val.do_return( 1);
            continue;
          }
        }
      }
      _np.execute_round_done(ITSIZE);
      __syncthreads();
    }
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
    p_propogation_flag[src] = true;
  }
  ret_val.thread_exit<_br>(_ts);
}
__global__ void PropFlagReset(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, bool * p_propogation_flag)
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
      p_propogation_flag[src] = false;
    }
  }
  // FP: "7 -> 8;
}
__global__ void SuccessorDecrement(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, unsigned int * p_num_successors, bool * p_propogation_flag, unsigned int * p_trim)
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
      if (p_num_successors[src] == 0 && !p_propogation_flag[src])
      {
        p_propogation_flag[src] = true;
        if (p_trim[src] > 0)
        {
          if (p_trim[src] > p_num_successors[src])
          {
            // DANGER
            //abort();
          }
          p_num_successors[src] -= p_trim[src];
          p_trim[src] = 0;
          if (p_num_successors[src] == 0)
          {
            p_propogation_flag[src] = false;
          }
        }
      }
    }
  }
  // FP: "19 -> 20;
}
__global__ void DependencyPropogation(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, const unsigned int  local_current_src_node, unsigned int * p_current_length, float * p_dependency, unsigned int * p_num_shortest_paths, unsigned int * p_num_successors, bool * p_propogation_flag, unsigned int * p_trim, Sum ret_val)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_DependencyPropogation;
  typedef cub::BlockReduce<int, TB_SIZE> _br;
  __shared__ _br::TempStorage _ts;
  ret_val.thread_entry();
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
    bool pop  = src < __end;
    if (pop)
    {
      if (graph.node_data[src] == local_current_src_node || p_num_successors[src] == 0)
      {
        // Loc added this in
        continue;
      }
    }
    struct NPInspector1 _np = {0,0,0,0,0,0};
    __shared__ struct { index_type src; } _np_closure [TB_SIZE];
    _np_closure[threadIdx.x].src = src;
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
      src = _np_closure[nps.tb.src].src;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type current_edge;
        current_edge = ns +_np_j;
        {
          index_type dst;
          unsigned int edge_weight;
          dst = graph.getAbsDestination(current_edge);
          edge_weight = 1;
          if (p_num_successors[dst] == 0 && !p_propogation_flag[dst])
          {
            if ((p_current_length[src] + edge_weight) == p_current_length[dst])
            {
              atomicAdd(&p_trim[src], (unsigned int)1);
              p_dependency[src] = p_dependency[src] + (((float)p_num_shortest_paths[src] / (float)p_num_shortest_paths[dst]) * (float)(1.0 + p_dependency[dst]));
              ret_val.do_return( 1);
              continue;
            }
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
        src = _np_closure[nps.warp.src[warpid]].src;
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type current_edge;
          current_edge = _np_w_start +_np_ii;
          {
            index_type dst;
            unsigned int edge_weight;
            dst = graph.getAbsDestination(current_edge);
            edge_weight = 1;
            if (p_num_successors[dst] == 0 && !p_propogation_flag[dst])
            {
              if ((p_current_length[src] + edge_weight) == p_current_length[dst])
              {
                atomicAdd(&p_trim[src], (unsigned int)1);
                p_dependency[src] = p_dependency[src] + (((float)p_num_shortest_paths[src] / (float)p_num_shortest_paths[dst]) * (float)(1.0 + p_dependency[dst]));
                ret_val.do_return( 1);
                continue;
              }
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
        index_type current_edge;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        current_edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          unsigned int edge_weight;
          dst = graph.getAbsDestination(current_edge);
          edge_weight = 1;
          if (p_num_successors[dst] == 0 && !p_propogation_flag[dst])
          {
            if ((p_current_length[src] + edge_weight) == p_current_length[dst])
            {
              atomicAdd(&p_trim[src], (unsigned int)1);
              p_dependency[src] = p_dependency[src] + (((float)p_num_shortest_paths[src] / (float)p_num_shortest_paths[dst]) * (float)(1.0 + p_dependency[dst]));
              ret_val.do_return( 1);
              continue;
            }
          }
        }
      }
      _np.execute_round_done(ITSIZE);
      __syncthreads();
    }
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
  }
  ret_val.thread_exit<_br>(_ts);
}
__global__ void BC(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, float * p_betweeness_centrality, float * p_dependency)
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
      p_betweeness_centrality[src] = p_betweeness_centrality[src] + p_dependency[src];
    }
  }
  // FP: "7 -> 8;
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
  InitializeGraph <<<blocks, threads>>>(ctx->gg, ctx->nowned, __begin, __end, ctx->betweeness_centrality.data.gpu_wr_ptr());
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
void InitializeIteration_cuda(unsigned int  __begin, unsigned int  __end, const unsigned int & local_infinity, const unsigned int & local_current_src_node, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  InitializeIteration <<<blocks, threads>>>(ctx->gg, ctx->nowned, __begin, __end, local_current_src_node, local_infinity, ctx->current_length.data.gpu_wr_ptr(), ctx->dependency.data.gpu_wr_ptr(), ctx->num_predecessors.data.gpu_wr_ptr(), ctx->num_shortest_paths.data.gpu_wr_ptr(), ctx->num_successors.data.gpu_wr_ptr(), ctx->old_length.data.gpu_wr_ptr(), ctx->propogation_flag.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void InitializeIteration_all_cuda(const unsigned int & local_infinity, const unsigned int & local_current_src_node, struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  InitializeIteration_cuda(0, ctx->nowned, local_infinity, local_current_src_node, ctx);
  // FP: "2 -> 3;
}
void FirstIterationSSSP_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  FirstIterationSSSP <<<blocks, __tb_FirstIterationSSSP>>>(ctx->gg, ctx->nowned, __begin, __end, ctx->current_length.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void FirstIterationSSSP_all_cuda(struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  FirstIterationSSSP_cuda(0, ctx->nowned, ctx);
  // FP: "2 -> 3;
}
void SSSP_cuda(unsigned int  __begin, unsigned int  __end, int & __retval, struct CUDA_Context * ctx)
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
  SSSP <<<blocks, __tb_SSSP>>>(ctx->gg, ctx->nowned, __begin, __end, ctx->current_length.data.gpu_wr_ptr(), ctx->old_length.data.gpu_wr_ptr(), _rv);
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
  __retval = *(retval.cpu_rd_ptr());
  // FP: "7 -> 8;
}
void SSSP_all_cuda(int & __retval, struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  SSSP_cuda(0, ctx->nowned, __retval, ctx);
  // FP: "2 -> 3;
}
void PredAndSucc_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  PredAndSucc <<<blocks, __tb_PredAndSucc>>>(ctx->gg, ctx->nowned, __begin, __end, ctx->current_length.data.gpu_wr_ptr(), ctx->num_predecessors.data.gpu_wr_ptr(), ctx->num_successors.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void PredAndSucc_all_cuda(struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  PredAndSucc_cuda(0, ctx->nowned, ctx);
  // FP: "2 -> 3;
}
void PredecessorDecrement_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  PredecessorDecrement <<<blocks, threads>>>(ctx->gg, ctx->nowned, __begin, __end, ctx->num_predecessors.data.gpu_wr_ptr(), ctx->propogation_flag.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void PredecessorDecrement_all_cuda(struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  PredecessorDecrement_cuda(0, ctx->nowned, ctx);
  // FP: "2 -> 3;
}
void NumShortestPaths_cuda(unsigned int  __begin, unsigned int  __end, int & __retval, struct CUDA_Context * ctx)
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
  NumShortestPaths <<<blocks, __tb_NumShortestPaths>>>(ctx->gg, ctx->nowned, __begin, __end, ctx->current_length.data.gpu_wr_ptr(), ctx->num_predecessors.data.gpu_wr_ptr(), ctx->num_shortest_paths.data.gpu_wr_ptr(), ctx->propogation_flag.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr(), _rv);
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
  __retval = *(retval.cpu_rd_ptr());
  // FP: "7 -> 8;
}
void NumShortestPaths_all_cuda(int & __retval, struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  NumShortestPaths_cuda(0, ctx->nowned, __retval, ctx);
  // FP: "2 -> 3;
}
void PropFlagReset_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  PropFlagReset <<<blocks, threads>>>(ctx->gg, ctx->nowned, __begin, __end, ctx->propogation_flag.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void PropFlagReset_all_cuda(struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  PropFlagReset_cuda(0, ctx->nowned, ctx);
  // FP: "2 -> 3;
}
void SuccessorDecrement_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  SuccessorDecrement <<<blocks, threads>>>(ctx->gg, ctx->nowned, __begin, __end, ctx->num_successors.data.gpu_wr_ptr(), ctx->propogation_flag.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void SuccessorDecrement_all_cuda(struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  SuccessorDecrement_cuda(0, ctx->nowned, ctx);
  // FP: "2 -> 3;
}
void DependencyPropogation_cuda(unsigned int  __begin, unsigned int  __end, int & __retval, const unsigned int & local_current_src_node, struct CUDA_Context * ctx)
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
  DependencyPropogation <<<blocks, __tb_DependencyPropogation>>>(ctx->gg, ctx->nowned, __begin, __end, local_current_src_node, ctx->current_length.data.gpu_wr_ptr(), ctx->dependency.data.gpu_wr_ptr(), ctx->num_shortest_paths.data.gpu_wr_ptr(), ctx->num_successors.data.gpu_wr_ptr(), ctx->propogation_flag.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr(), _rv);
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
  __retval = *(retval.cpu_rd_ptr());
  // FP: "7 -> 8;
}
void DependencyPropogation_all_cuda(int & __retval, const unsigned int & local_current_src_node, struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  DependencyPropogation_cuda(0, ctx->nowned, __retval, local_current_src_node, ctx);
  // FP: "2 -> 3;
}
void BC_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  BC <<<blocks, threads>>>(ctx->gg, ctx->nowned, __begin, __end, ctx->betweeness_centrality.data.gpu_wr_ptr(), ctx->dependency.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void BC_all_cuda(struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  BC_cuda(0, ctx->nowned, ctx);
  // FP: "2 -> 3;
}
