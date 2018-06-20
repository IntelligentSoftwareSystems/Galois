/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
#include "kernels/reduce.cuh"
#include "kcore_pull_cuda.cuh"
static const int __tb_KCore = TB_SIZE;
__global__ void DegreeCounting(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_current_degree, DynamicBitset& bitset_current_degree)
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
      p_current_degree[src] = graph.getOutDegree(src);
      bitset_current_degree.set(src);
    }
  }
  // FP: "8 -> 9;
}
__global__ void InitializeGraph(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_current_degree, uint8_t * p_flag, uint8_t * p_pull_flag, uint32_t * p_trim)
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
      p_pull_flag[src] = false;
    }
  }
  // FP: "10 -> 11;
}
__global__ void LiveUpdate(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t local_k_core_num, uint32_t * p_current_degree, uint8_t * p_flag, uint8_t * p_pull_flag, uint32_t * p_trim, HGAccumulator<unsigned int> DGAccumulator_accum)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  __shared__ cub::BlockReduce<unsigned int, TB_SIZE>::TempStorage DGAccumulator_accum_ts;
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
        if (p_trim[src] > 0)
        {
          p_current_degree[src] = p_current_degree[src] - p_trim[src];
        }
        if (p_current_degree[src] < local_k_core_num)
        {
          p_flag[src] = false;
          DGAccumulator_accum.reduce( 1);
          p_pull_flag[src] = true;
        }
      }
      else
      {
        if (p_pull_flag[src])
        {
          p_pull_flag[src] = false;
        }
      }
      p_trim[src] = 0;
    }
  }
  // FP: "22 -> 23;
  DGAccumulator_accum.thread_exit<cub::BlockReduce<unsigned int, TB_SIZE> >(DGAccumulator_accum_ts);
  // FP: "23 -> 24;
}
__global__ void KCore(CSRGraph graph, unsigned int __begin, unsigned int __end, uint8_t * p_flag, uint8_t * p_pull_flag, uint32_t * p_trim, DynamicBitset& bitset_trim)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_KCore;
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
      if (p_flag[src])
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
          dst = graph.getAbsDestination(current_edge);
          if (p_pull_flag[dst])
          {
            atomicTestAdd(&p_trim[src], (uint32_t)1);
            bitset_trim.set(src);
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
          index_type current_edge;
          current_edge = _np_w_start +_np_ii;
          {
            index_type dst;
            dst = graph.getAbsDestination(current_edge);
            if (p_pull_flag[dst])
            {
              atomicTestAdd(&p_trim[src], (uint32_t)1);
              bitset_trim.set(src);
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
        index_type current_edge;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        current_edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          dst = graph.getAbsDestination(current_edge);
          if (p_pull_flag[dst])
          {
            atomicTestAdd(&p_trim[src], (uint32_t)1);
            bitset_trim.set(src);
          }
        }
      }
      // FP: "101 -> 102;
      _np.execute_round_done(ITSIZE);
      // FP: "102 -> 103;
      __syncthreads();
    }
    // FP: "104 -> 105;
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
  }
  // FP: "106 -> 107;
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
void DegreeCounting_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  DegreeCounting <<<blocks, threads>>>(ctx->gg, __begin, __end, ctx->current_degree.data.gpu_wr_ptr(), *(ctx->current_degree.is_updated.gpu_rd_ptr()));
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void DegreeCounting_allNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  DegreeCounting_cuda(0, ctx->gg.nnodes, ctx);
  // FP: "2 -> 3;
}
void DegreeCounting_masterNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  DegreeCounting_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx);
  // FP: "2 -> 3;
}
void DegreeCounting_nodesWithEdges_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  DegreeCounting_cuda(0, ctx->numNodesWithEdges, ctx);
  // FP: "2 -> 3;
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
  InitializeGraph <<<blocks, threads>>>(ctx->gg, __begin, __end, ctx->current_degree.data.gpu_wr_ptr(), ctx->flag.data.gpu_wr_ptr(), ctx->pull_flag.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr());
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
void LiveUpdate_cuda(unsigned int  __begin, unsigned int  __end, unsigned int & DGAccumulator_accum, uint32_t local_k_core_num, struct CUDA_Context*  ctx)
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
  LiveUpdate <<<blocks, threads>>>(ctx->gg, __begin, __end, local_k_core_num, ctx->current_degree.data.gpu_wr_ptr(), ctx->flag.data.gpu_wr_ptr(), ctx->pull_flag.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr(), _DGAccumulator_accum);
  // FP: "9 -> 10;
  check_cuda_kernel;
  // FP: "10 -> 11;
  DGAccumulator_accum = *(DGAccumulator_accumval.cpu_rd_ptr());
  // FP: "11 -> 12;
}
void LiveUpdate_allNodes_cuda(unsigned int & DGAccumulator_accum, uint32_t local_k_core_num, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  LiveUpdate_cuda(0, ctx->gg.nnodes, DGAccumulator_accum, local_k_core_num, ctx);
  // FP: "2 -> 3;
}
void LiveUpdate_masterNodes_cuda(unsigned int & DGAccumulator_accum, uint32_t local_k_core_num, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  LiveUpdate_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, DGAccumulator_accum, local_k_core_num, ctx);
  // FP: "2 -> 3;
}
void LiveUpdate_nodesWithEdges_cuda(unsigned int & DGAccumulator_accum, uint32_t local_k_core_num, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  LiveUpdate_cuda(0, ctx->numNodesWithEdges, DGAccumulator_accum, local_k_core_num, ctx);
  // FP: "2 -> 3;
}
void KCore_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  KCore <<<blocks, __tb_KCore>>>(ctx->gg, __begin, __end, ctx->flag.data.gpu_wr_ptr(), ctx->pull_flag.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr(), *(ctx->trim.is_updated.gpu_rd_ptr()));
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void KCore_allNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  KCore_cuda(0, ctx->gg.nnodes, ctx);
  // FP: "2 -> 3;
}
void KCore_masterNodes_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  KCore_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, ctx);
  // FP: "2 -> 3;
}
void KCore_nodesWithEdges_cuda(struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  KCore_cuda(0, ctx->numNodesWithEdges, ctx);
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