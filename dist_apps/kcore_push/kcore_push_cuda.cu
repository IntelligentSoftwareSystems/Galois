/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
#include "kernels/reduce.cuh"
#include "kcore_push_cuda.cuh"
static const int __tb_InitializeGraph2 = TB_SIZE;
static const int __tb_KCoreStep1 = TB_SIZE;
__global__ void InitializeGraph2(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_current_degree, DynamicBitset& bitset_current_degree)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_InitializeGraph2;
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
    __shared__ struct { ; } _np_closure [TB_SIZE];
    // FP: "14 -> 15;
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
      // FP: "43 -> 44;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type current_edge;
        current_edge = ns +_np_j;
        {
          index_type dest_node;
          dest_node = graph.getAbsDestination(current_edge);
          atomicTestAdd(&p_current_degree[dest_node], (uint32_t)1);
          bitset_current_degree.set(dest_node);
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
          index_type current_edge;
          current_edge = _np_w_start +_np_ii;
          {
            index_type dest_node;
            dest_node = graph.getAbsDestination(current_edge);
            atomicTestAdd(&p_current_degree[dest_node], (uint32_t)1);
            bitset_current_degree.set(dest_node);
          }
        }
      }
      // FP: "74 -> 75;
      __syncthreads();
      // FP: "75 -> 76;
    }

    // FP: "76 -> 77;
    __syncthreads();
    // FP: "77 -> 78;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "78 -> 79;
    while (_np.work())
    {
      // FP: "79 -> 80;
      int _np_i =0;
      // FP: "80 -> 81;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "81 -> 82;
      __syncthreads();
      // FP: "82 -> 83;

      // FP: "83 -> 84;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type current_edge;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        current_edge= nps.fg.itvalue[_np_i];
        {
          index_type dest_node;
          dest_node = graph.getAbsDestination(current_edge);
          atomicTestAdd(&p_current_degree[dest_node], (uint32_t)1);
          bitset_current_degree.set(dest_node);
        }
      }
      // FP: "92 -> 93;
      _np.execute_round_done(ITSIZE);
      // FP: "93 -> 94;
      __syncthreads();
    }
    // FP: "95 -> 96;
    assert(threadIdx.x < __kernel_tb_size);
  }
  // FP: "97 -> 98;
}
__global__ void InitializeGraph1(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_current_degree, uint8_t * p_flag, uint32_t * p_trim)
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
      p_flag[src] = true;
      p_trim[src] = 0;
      p_current_degree[src] = 0;
    }
  }
  // FP: "9 -> 10;
}
__global__ void KCoreStep2(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_current_degree, uint8_t * p_flag, uint32_t * p_trim)
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
      if (p_flag[src])
      {
        if (p_trim[src] > 0)
        {
          p_current_degree[src] = p_current_degree[src] - p_trim[src];
        }
      }
      p_trim[src] = 0;
    }
  }
  // FP: "12 -> 13;
}
__global__ void KCoreStep1(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t local_k_core_num, uint32_t * p_current_degree, uint8_t * p_flag, uint32_t * p_trim, DynamicBitset& bitset_trim, HGAccumulator<unsigned int> DGAccumulator_accum)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_KCoreStep1;
  __shared__ cub::BlockReduce<unsigned int, TB_SIZE>::TempStorage DGAccumulator_accum_ts;
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
      if (p_flag[src])
      {
        if (p_current_degree[src] < local_k_core_num)
        {
          p_flag[src] = false;
          DGAccumulator_accum.reduce( 1);
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
    // FP: "17 -> 18;
    // FP: "20 -> 21;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "21 -> 22;
    __shared__ struct { ; } _np_closure [TB_SIZE];
    // FP: "22 -> 23;
    // FP: "23 -> 24;
    if (pop)
    {
      _np.size = (graph).getOutDegree(src);
      _np.start = (graph).getFirstEdge(src);
    }
    // FP: "26 -> 27;
    // FP: "27 -> 28;
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    // FP: "28 -> 29;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    // FP: "29 -> 30;
    if (threadIdx.x == 0)
    {
      nps.tb.owner = MAX_TB_SIZE + 1;
    }
    // FP: "32 -> 33;
    __syncthreads();
    // FP: "33 -> 34;
    while (true)
    {
      // FP: "34 -> 35;
      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }
      // FP: "37 -> 38;
      __syncthreads();
      // FP: "38 -> 39;
      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {
        // FP: "39 -> 40;
        __syncthreads();
        // FP: "40 -> 41;
        break;
      }
      // FP: "42 -> 43;
      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }
      // FP: "45 -> 46;
      __syncthreads();
      // FP: "46 -> 47;
      int ns = nps.tb.start;
      int ne = nps.tb.size;
      // FP: "47 -> 48;
      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }
      // FP: "50 -> 51;
      assert(nps.tb.src < __kernel_tb_size);
      // FP: "51 -> 52;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type current_edge;
        current_edge = ns +_np_j;
        {
          index_type dst;
          dst = graph.getAbsDestination(current_edge);
          atomicTestAdd(&p_trim[dst], (uint32_t)1);
          bitset_trim.set(dst);
        }
      }
      // FP: "59 -> 60;
      __syncthreads();
    }
    // FP: "61 -> 62;

    // FP: "62 -> 63;
    {
      const int warpid = threadIdx.x / 32;
      // FP: "63 -> 64;
      const int _np_laneid = cub::LaneId();
      // FP: "64 -> 65;
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
          index_type current_edge;
          current_edge = _np_w_start +_np_ii;
          {
            index_type dst;
            dst = graph.getAbsDestination(current_edge);
            atomicTestAdd(&p_trim[dst], (uint32_t)1);
            bitset_trim.set(dst);
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
        current_edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          dst = graph.getAbsDestination(current_edge);
          atomicTestAdd(&p_trim[dst], (uint32_t)1);
          bitset_trim.set(dst);
        }
      }
      // FP: "100 -> 101;
      _np.execute_round_done(ITSIZE);
      // FP: "101 -> 102;
      __syncthreads();
    }
    // FP: "103 -> 104;
    assert(threadIdx.x < __kernel_tb_size);
  }
  // FP: "107 -> 108;
  DGAccumulator_accum.thread_exit<cub::BlockReduce<unsigned int, TB_SIZE> >(DGAccumulator_accum_ts);
  // FP: "108 -> 109;
}
__global__ void KCoreSanityCheck(CSRGraph graph, unsigned int __begin, unsigned int __end, uint8_t * p_flag, HGAccumulator<uint64_t> DGAccumulator_accum)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  __shared__ cub::BlockReduce<uint64_t, TB_SIZE>::TempStorage DGAccumulator_accum_ts;
  index_type src_end;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  DGAccumulator_accum.thread_entry();
  // FP: "3 -> 4;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      if (p_flag[src])
      {
        DGAccumulator_accum.reduce( 1);
      }
    }
  }
  // FP: "11 -> 12;
  DGAccumulator_accum.thread_exit<cub::BlockReduce<uint64_t, TB_SIZE> >(DGAccumulator_accum_ts);
  // FP: "12 -> 13;
}
void InitializeGraph2_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  InitializeGraph2 <<<blocks, __tb_InitializeGraph2>>>(ctx->gg, __begin, __end, ctx->current_degree.data.gpu_wr_ptr(), *(ctx->current_degree.is_updated.gpu_rd_ptr()));
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void InitializeGraph2_allNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  InitializeGraph2_cuda(0, ctx->gg.nnodes, ctx);
  // FP: "2 -> 3;
}
void InitializeGraph2_masterNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  InitializeGraph2_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx);
  // FP: "2 -> 3;
}
void InitializeGraph2_nodesWithEdges_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  InitializeGraph2_cuda(0, ctx->numNodesWithEdges, ctx);
  // FP: "2 -> 3;
}
void InitializeGraph1_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  InitializeGraph1 <<<blocks, threads>>>(ctx->gg, __begin, __end, ctx->current_degree.data.gpu_wr_ptr(), ctx->flag.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void InitializeGraph1_allNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  InitializeGraph1_cuda(0, ctx->gg.nnodes, ctx);
  // FP: "2 -> 3;
}
void InitializeGraph1_masterNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  InitializeGraph1_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx);
  // FP: "2 -> 3;
}
void InitializeGraph1_nodesWithEdges_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  InitializeGraph1_cuda(0, ctx->numNodesWithEdges, ctx);
  // FP: "2 -> 3;
}
void KCoreStep2_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  KCoreStep2 <<<blocks, threads>>>(ctx->gg, __begin, __end, ctx->current_degree.data.gpu_wr_ptr(), ctx->flag.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void KCoreStep2_allNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  KCoreStep2_cuda(0, ctx->gg.nnodes, ctx);
  // FP: "2 -> 3;
}
void KCoreStep2_masterNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  KCoreStep2_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx);
  // FP: "2 -> 3;
}
void KCoreStep2_nodesWithEdges_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  KCoreStep2_cuda(0, ctx->numNodesWithEdges, ctx);
  // FP: "2 -> 3;
}
void KCoreStep1_cuda(unsigned int  __begin, unsigned int  __end, unsigned int & DGAccumulator_accum, uint32_t local_k_core_num, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  HGAccumulator<unsigned int> _DGAccumulator_accum;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  Shared<unsigned int> DGAccumulator_accumval  = Shared<unsigned int>(1);
  // FP: "5 -> 6;
  // FP: "6 -> 7;
  *(DGAccumulator_accumval.cpu_wr_ptr()) = 0;
  // FP: "7 -> 8;
  _DGAccumulator_accum.rv = DGAccumulator_accumval.gpu_wr_ptr();
  // FP: "8 -> 9;
  KCoreStep1 <<<blocks, __tb_KCoreStep1>>>(ctx->gg, __begin, __end, local_k_core_num, ctx->current_degree.data.gpu_wr_ptr(), ctx->flag.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr(), *(ctx->trim.is_updated.gpu_rd_ptr()), _DGAccumulator_accum);
  // FP: "9 -> 10;
  check_cuda_kernel;
  // FP: "10 -> 11;
  DGAccumulator_accum = *(DGAccumulator_accumval.cpu_rd_ptr());
  // FP: "11 -> 12;
}
void KCoreStep1_allNodes_cuda(unsigned int & DGAccumulator_accum, uint32_t local_k_core_num, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  KCoreStep1_cuda(0, ctx->gg.nnodes, DGAccumulator_accum, local_k_core_num, ctx);
  // FP: "2 -> 3;
}
void KCoreStep1_masterNodes_cuda(unsigned int & DGAccumulator_accum, uint32_t local_k_core_num, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  KCoreStep1_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, DGAccumulator_accum, local_k_core_num, ctx);
  // FP: "2 -> 3;
}
void KCoreStep1_nodesWithEdges_cuda(unsigned int & DGAccumulator_accum, uint32_t local_k_core_num, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  KCoreStep1_cuda(0, ctx->numNodesWithEdges, DGAccumulator_accum, local_k_core_num, ctx);
  // FP: "2 -> 3;
}
void KCoreSanityCheck_cuda(unsigned int  __begin, unsigned int  __end, uint64_t & DGAccumulator_accum, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  HGAccumulator<uint64_t> _DGAccumulator_accum;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  Shared<uint64_t> DGAccumulator_accumval  = Shared<uint64_t>(1);
  // FP: "5 -> 6;
  // FP: "6 -> 7;
  *(DGAccumulator_accumval.cpu_wr_ptr()) = 0;
  // FP: "7 -> 8;
  _DGAccumulator_accum.rv = DGAccumulator_accumval.gpu_wr_ptr();
  // FP: "8 -> 9;
  KCoreSanityCheck <<<blocks, threads>>>(ctx->gg, __begin, __end, ctx->flag.data.gpu_wr_ptr(), _DGAccumulator_accum);
  // FP: "9 -> 10;
  check_cuda_kernel;
  // FP: "10 -> 11;
  DGAccumulator_accum = *(DGAccumulator_accumval.cpu_rd_ptr());
  // FP: "11 -> 12;
}
void KCoreSanityCheck_allNodes_cuda(uint64_t & DGAccumulator_accum, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  KCoreSanityCheck_cuda(0, ctx->gg.nnodes, DGAccumulator_accum, ctx);
  // FP: "2 -> 3;
}
void KCoreSanityCheck_masterNodes_cuda(uint64_t & DGAccumulator_accum, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  KCoreSanityCheck_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, DGAccumulator_accum, ctx);
  // FP: "2 -> 3;
}
void KCoreSanityCheck_nodesWithEdges_cuda(uint64_t & DGAccumulator_accum, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  KCoreSanityCheck_cuda(0, ctx->numNodesWithEdges, DGAccumulator_accum, ctx);
  // FP: "2 -> 3;
}