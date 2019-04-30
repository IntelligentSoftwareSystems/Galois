/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
#include "kernels/reduce.cuh"
#include "sssp_pull_cuda.cuh"
static const int __tb_SSSP = TB_SIZE;
__global__ void InitializeGraph(CSRGraph graph, unsigned int __begin, unsigned int __end, const uint32_t  local_infinity, unsigned long long local_src_node, uint32_t * p_dist_current)
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
__global__ void SSSP(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_dist_current, DynamicBitset& bitset_dist_current, HGAccumulator<unsigned int> active_vertices)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_SSSP;
  __shared__ cub::BlockReduce<unsigned int, TB_SIZE>::TempStorage active_vertices_ts;
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
  active_vertices.thread_entry();
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
    }
    // FP: "11 -> 12;
    // FP: "14 -> 15;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "15 -> 16;
    __shared__ struct { index_type src; } _np_closure [TB_SIZE];
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
          uint32_t new_dist;
          uint32_t old_dist;
          dst = graph.getAbsDestination(jj);
          new_dist = p_dist_current[dst] + graph.getAbsWeight(jj);
          old_dist = atomicTestMin(&p_dist_current[src], new_dist);
          if (old_dist > new_dist)
          {
            bitset_dist_current.set(src);
            active_vertices.reduce( 1);
          }
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
            new_dist = p_dist_current[dst] + graph.getAbsWeight(jj);
            old_dist = atomicTestMin(&p_dist_current[src], new_dist);
            if (old_dist > new_dist)
            {
              bitset_dist_current.set(src);
              active_vertices.reduce( 1);
            }
          }
        }
      }
      // FP: "88 -> 89;
      __syncthreads();
      // FP: "89 -> 90;
    }

    // FP: "90 -> 91;
    __syncthreads();
    // FP: "91 -> 92;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "92 -> 93;
    while (_np.work())
    {
      // FP: "93 -> 94;
      int _np_i =0;
      // FP: "94 -> 95;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "95 -> 96;
      __syncthreads();
      // FP: "96 -> 97;

      // FP: "97 -> 98;
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
          new_dist = p_dist_current[dst] + graph.getAbsWeight(jj);
          old_dist = atomicTestMin(&p_dist_current[src], new_dist);
          if (old_dist > new_dist)
          {
            bitset_dist_current.set(src);
            active_vertices.reduce( 1);
          }
        }
      }
      // FP: "112 -> 113;
      _np.execute_round_done(ITSIZE);
      // FP: "113 -> 114;
      __syncthreads();
    }
    // FP: "115 -> 116;
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
  }
  // FP: "117 -> 118;
  active_vertices.thread_exit<cub::BlockReduce<unsigned int, TB_SIZE> >(active_vertices_ts);
  // FP: "118 -> 119;
}
__global__ void SSSPSanityCheck(CSRGraph graph, unsigned int __begin, unsigned int __end, const uint32_t  local_infinity, uint32_t * p_dist_current, HGAccumulator<uint64_t> DGAccumulator_sum, HGAccumulator<uint64_t> dg_avg, HGReduceMax<uint32_t> DGMax)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  __shared__ cub::BlockReduce<uint64_t, TB_SIZE>::TempStorage DGAccumulator_sum_ts;
  __shared__ cub::BlockReduce<uint64_t, TB_SIZE>::TempStorage dg_avg_ts;
  __shared__ cub::BlockReduce<uint32_t, TB_SIZE>::TempStorage DGMax_ts;
  index_type src_end;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  DGAccumulator_sum.thread_entry();
  // FP: "3 -> 4;
  // FP: "4 -> 5;
  dg_avg.thread_entry();
  // FP: "5 -> 6;
  // FP: "6 -> 7;
  DGMax.thread_entry();
  // FP: "7 -> 8;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      if (p_dist_current[src] < local_infinity)
      {
        DGAccumulator_sum.reduce( 1);
        DGMax.reduce(p_dist_current[src]);
        dg_avg.reduce( p_dist_current[src]);
      }
    }
  }
  // FP: "17 -> 18;
  DGAccumulator_sum.thread_exit<cub::BlockReduce<uint64_t, TB_SIZE> >(DGAccumulator_sum_ts);
  // FP: "18 -> 19;
  dg_avg.thread_exit<cub::BlockReduce<uint64_t, TB_SIZE> >(dg_avg_ts);
  // FP: "19 -> 20;
  DGMax.thread_exit<cub::BlockReduce<uint32_t, TB_SIZE> >(DGMax_ts);
  // FP: "20 -> 21;
}
void InitializeGraph_cuda(unsigned int  __begin, unsigned int  __end, const uint32_t & local_infinity, unsigned long long local_src_node, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  InitializeGraph <<<blocks, threads>>>(ctx->gg, __begin, __end, local_infinity, local_src_node, ctx->dist_current.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void InitializeGraph_allNodes_cuda(const uint32_t & local_infinity, unsigned long long local_src_node, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  InitializeGraph_cuda(0, ctx->gg.nnodes, local_infinity, local_src_node, ctx);
  // FP: "2 -> 3;
}
void InitializeGraph_masterNodes_cuda(const uint32_t & local_infinity, unsigned long long local_src_node, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  InitializeGraph_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_infinity, local_src_node, ctx);
  // FP: "2 -> 3;
}
void InitializeGraph_nodesWithEdges_cuda(const uint32_t & local_infinity, unsigned long long local_src_node, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  InitializeGraph_cuda(0, ctx->numNodesWithEdges, local_infinity, local_src_node, ctx);
  // FP: "2 -> 3;
}
void SSSP_cuda(unsigned int  __begin, unsigned int  __end, unsigned int & active_vertices, struct CUDA_Context*  ctx)
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
  SSSP <<<blocks, __tb_SSSP>>>(ctx->gg, __begin, __end, ctx->dist_current.data.gpu_wr_ptr(), *(ctx->dist_current.is_updated.gpu_rd_ptr()), _active_vertices);
  // FP: "9 -> 10;
  check_cuda_kernel;
  // FP: "10 -> 11;
  active_vertices = *(active_verticesval.cpu_rd_ptr());
  // FP: "11 -> 12;
}
void SSSP_allNodes_cuda(unsigned int & active_vertices, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  SSSP_cuda(0, ctx->gg.nnodes, active_vertices, ctx);
  // FP: "2 -> 3;
}
void SSSP_masterNodes_cuda(unsigned int & active_vertices, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  SSSP_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, active_vertices, ctx);
  // FP: "2 -> 3;
}
void SSSP_nodesWithEdges_cuda(unsigned int & active_vertices, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  SSSP_cuda(0, ctx->numNodesWithEdges, active_vertices, ctx);
  // FP: "2 -> 3;
}
void SSSPSanityCheck_cuda(unsigned int  __begin, unsigned int  __end, uint64_t & DGAccumulator_sum, uint64_t & dg_avg, uint32_t & DGMax, const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  HGAccumulator<uint64_t> _DGAccumulator_sum;
  HGAccumulator<uint64_t> _dg_avg;
  HGReduceMax<uint32_t> _DGMax;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  Shared<uint64_t> DGAccumulator_sumval  = Shared<uint64_t>(1);
  // FP: "5 -> 6;
  // FP: "6 -> 7;
  *(DGAccumulator_sumval.cpu_wr_ptr()) = 0;
  // FP: "7 -> 8;
  _DGAccumulator_sum.rv = DGAccumulator_sumval.gpu_wr_ptr();
  // FP: "8 -> 9;
  Shared<uint64_t> dg_avgval  = Shared<uint64_t>(1);
  // FP: "9 -> 10;
  // FP: "10 -> 11;
  *(dg_avgval.cpu_wr_ptr()) = 0;
  // FP: "11 -> 12;
  _dg_avg.rv = dg_avgval.gpu_wr_ptr();
  // FP: "12 -> 13;
  Shared<uint32_t> DGMaxval  = Shared<uint32_t>(1);
  // FP: "13 -> 14;
  // FP: "14 -> 15;
  *(DGMaxval.cpu_wr_ptr()) = 0;
  // FP: "15 -> 16;
  _DGMax.rv = DGMaxval.gpu_wr_ptr();
  // FP: "16 -> 17;
  SSSPSanityCheck <<<blocks, threads>>>(ctx->gg, __begin, __end, local_infinity, ctx->dist_current.data.gpu_wr_ptr(), _DGAccumulator_sum, _dg_avg, _DGMax);
  // FP: "17 -> 18;
  check_cuda_kernel;
  // FP: "18 -> 19;
  DGAccumulator_sum = *(DGAccumulator_sumval.cpu_rd_ptr());
  // FP: "19 -> 20;
  dg_avg = *(dg_avgval.cpu_rd_ptr());
  // FP: "20 -> 21;
  DGMax = *(DGMaxval.cpu_rd_ptr());
  // FP: "21 -> 22;
}
void SSSPSanityCheck_allNodes_cuda(uint64_t & DGAccumulator_sum, uint64_t & dg_avg, uint32_t & DGMax, const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  SSSPSanityCheck_cuda(0, ctx->gg.nnodes, DGAccumulator_sum, dg_avg, DGMax, local_infinity, ctx);
  // FP: "2 -> 3;
}
void SSSPSanityCheck_masterNodes_cuda(uint64_t & DGAccumulator_sum, uint64_t & dg_avg, uint32_t & DGMax, const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  SSSPSanityCheck_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, DGAccumulator_sum, dg_avg, DGMax, local_infinity, ctx);
  // FP: "2 -> 3;
}
void SSSPSanityCheck_nodesWithEdges_cuda(uint64_t & DGAccumulator_sum, uint64_t & dg_avg, uint32_t & DGMax, const uint32_t & local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  SSSPSanityCheck_cuda(0, ctx->numNodesWithEdges, DGAccumulator_sum, dg_avg, DGMax, local_infinity, ctx);
  // FP: "2 -> 3;
}