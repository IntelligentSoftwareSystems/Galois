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
uint32_t * P_COMP_CURRENT;
uint32_t * P_COMP_OLD;
#include "gen_cuda.cuh"
static const int __tb_ConnectedComp = TB_SIZE;
static const int __tb_FirstItr_ConnectedComp = TB_SIZE;
__global__ void InitializeGraph(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, uint32_t * p_comp_current, uint32_t * p_comp_old)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  index_type src_end;
  // FP: "1 -> 2;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      p_comp_current[src] = graph.node_data[src];
      p_comp_old[src] = graph.node_data[src];
    }
  }
  // FP: "8 -> 9;
}
__global__ void FirstItr_ConnectedComp(CSRGraph graph, DynamicBitset *is_updated, unsigned int __nowned, unsigned int __begin, unsigned int __end, uint32_t * p_comp_current, uint32_t * p_comp_old)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_FirstItr_ConnectedComp;
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
      p_comp_old[src] = p_comp_current[src];
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
          dst = graph.getAbsDestination(jj);
          new_dist = p_comp_current[src];
          if (p_comp_current[dst] > new_dist) {
            uint32_t old_dist = atomicMin(&p_comp_current[dst], new_dist);
            if (old_dist > new_dist) is_updated->set(dst);
          }
        }
      }
      // FP: "53 -> 54;
      __syncthreads();
    }
    // FP: "55 -> 56;

    // FP: "56 -> 57;
    {
      const int warpid = threadIdx.x / 32;
      // FP: "57 -> 58;
      const int _np_laneid = cub::LaneId();
      // FP: "58 -> 59;
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
          index_type jj;
          jj = _np_w_start +_np_ii;
          {
            index_type dst;
            uint32_t new_dist;
            dst = graph.getAbsDestination(jj);
            new_dist = p_comp_current[src];
            if (p_comp_current[dst] > new_dist) {
              uint32_t old_dist = atomicMin(&p_comp_current[dst], new_dist);
              if (old_dist > new_dist) is_updated->set(dst);
            }
          }
        }
      }
      // FP: "77 -> 78;
      __syncthreads();
      // FP: "78 -> 79;
    }

    // FP: "79 -> 80;
    __syncthreads();
    // FP: "80 -> 81;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "81 -> 82;
    while (_np.work())
    {
      // FP: "82 -> 83;
      int _np_i =0;
      // FP: "83 -> 84;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "84 -> 85;
      __syncthreads();
      // FP: "85 -> 86;

      // FP: "86 -> 87;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type jj;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        jj= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          uint32_t new_dist;
          dst = graph.getAbsDestination(jj);
          new_dist = p_comp_current[src];
          if (p_comp_current[dst] > new_dist) {
            uint32_t old_dist = atomicMin(&p_comp_current[dst], new_dist);
            if (old_dist > new_dist) is_updated->set(dst);
          }
        }
      }
      // FP: "96 -> 97;
      _np.execute_round_done(ITSIZE);
      // FP: "97 -> 98;
      __syncthreads();
    }
    // FP: "99 -> 100;
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
  }
  // FP: "101 -> 102;
}
__global__ void ConnectedComp(CSRGraph graph, DynamicBitset *is_updated, unsigned int __nowned, unsigned int __begin, unsigned int __end, uint32_t * p_comp_current, uint32_t * p_comp_old, HGAccumulator<int> ret_val)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_ConnectedComp;
  typedef cub::BlockReduce<int, TB_SIZE> _br;
  __shared__ _br::TempStorage _ts;
  ret_val.thread_entry();
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
      if (p_comp_old[src] > p_comp_current[src])
      {
        p_comp_old[src] = p_comp_current[src];
        ret_val.reduce( 1);
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
        index_type jj;
        jj = ns +_np_j;
        {
          index_type dst;
          uint32_t new_dist;
          dst = graph.getAbsDestination(jj);
          new_dist = p_comp_current[src];
          if (p_comp_current[dst] > new_dist) {
            uint32_t old_dist = atomicMin(&p_comp_current[dst], new_dist);
            if (old_dist > new_dist) is_updated->set(dst);
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
          index_type jj;
          jj = _np_w_start +_np_ii;
          {
            index_type dst;
            uint32_t new_dist;
            dst = graph.getAbsDestination(jj);
            new_dist = p_comp_current[src];
            if (p_comp_current[dst] > new_dist) {
              uint32_t old_dist = atomicMin(&p_comp_current[dst], new_dist);
              if (old_dist > new_dist) is_updated->set(dst);
            }
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
        index_type jj;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        jj= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          uint32_t new_dist;
          dst = graph.getAbsDestination(jj);
          new_dist = p_comp_current[src];
          if (p_comp_current[dst] > new_dist) {
            uint32_t old_dist = atomicMin(&p_comp_current[dst], new_dist);
            if (old_dist > new_dist) is_updated->set(dst);
          }
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
    // FP: "103 -> 104;
  }
  ret_val.thread_exit<_br>(_ts);
}
__global__ void ConnectedCompSanityCheck(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_comp_current, HGAccumulator<unsigned int> sum)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;
  typedef cub::BlockReduce<unsigned int, TB_SIZE> _br;
  __shared__ _br::TempStorage _ts;
  sum.thread_entry();
  for (index_type src = __begin + tid; src < __end; src += nthreads)
  {
    if (p_comp_current[src] == graph.node_data[src]) {
      sum.reduce(1);
    }
  }
  sum.thread_exit<_br>(_ts);
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
  InitializeGraph <<<blocks, threads>>>(ctx->gg, ctx->numNodesWithEdges, __begin, __end, ctx->comp_current.data.gpu_wr_ptr(), ctx->comp_old.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void InitializeGraph_all_cuda(struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  InitializeGraph_cuda(0, ctx->numNodesWithEdges, ctx);
  // FP: "2 -> 3;
}
void FirstItr_ConnectedComp_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  FirstItr_ConnectedComp <<<blocks, __tb_FirstItr_ConnectedComp>>>(ctx->gg, ctx->comp_current.is_updated.gpu_rd_ptr(), ctx->numNodesWithEdges, __begin, __end, ctx->comp_current.data.gpu_wr_ptr(), ctx->comp_old.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void FirstItr_ConnectedComp_all_cuda(struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  FirstItr_ConnectedComp_cuda(0, ctx->numNodesWithEdges, ctx);
  // FP: "2 -> 3;
}
void ConnectedComp_cuda(unsigned int  __begin, unsigned int  __end, int & __retval, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  Shared<int> retval = Shared<int>(1);
  HGAccumulator<int> _rv;
  *(retval.cpu_wr_ptr()) = 0;
  _rv.rv = retval.gpu_wr_ptr();
  ConnectedComp <<<blocks, __tb_ConnectedComp>>>(ctx->gg, ctx->comp_current.is_updated.gpu_rd_ptr(), ctx->numNodesWithEdges, __begin, __end, ctx->comp_current.data.gpu_wr_ptr(), ctx->comp_old.data.gpu_wr_ptr(), _rv);
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
  __retval = *(retval.cpu_rd_ptr());
  // FP: "7 -> 8;
}
void ConnectedComp_all_cuda(int & __retval, struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  ConnectedComp_cuda(0, ctx->numNodesWithEdges, __retval, ctx);
  // FP: "2 -> 3;
}
void ConnectedCompSanityCheck_cuda(unsigned int & sum, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);
  Shared<unsigned int> sumval = Shared<unsigned int>(1);
  HGAccumulator<unsigned int> _sum;
  *(sumval.cpu_wr_ptr()) = 0;
  _sum.rv = sumval.gpu_wr_ptr();
  ConnectedCompSanityCheck <<<blocks, __tb_ConnectedComp>>>(ctx->gg, ctx->beginMaster, ctx->beginMaster+ctx->numOwned, ctx->comp_current.data.gpu_rd_ptr(), _sum);
  check_cuda_kernel;
  sum = *(sumval.cpu_rd_ptr());
}
