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
      p_flag[src]           = true;
      p_trim[src]           = 0;
      p_current_degree[src] = 0;
      p_pull_flag[src]      = false;
    }
  }
  // FP: "10 -> 11;
}
__global__ void LiveUpdate(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t local_k_core_num, uint32_t * p_current_degree, uint8_t * p_flag, uint8_t * p_pull_flag, uint32_t * p_trim, HGAccumulator<unsigned int> active_vertices)
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
      if (p_flag[src])
      {
        if (p_trim[src] > 0)
        {
          p_current_degree[src] = p_current_degree[src] - p_trim[src];
        }
        if (p_current_degree[src] < local_k_core_num)
        {
          p_flag[src] = false;
          active_vertices.reduce( 1);
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
  active_vertices.thread_exit<cub::BlockReduce<unsigned int, TB_SIZE> >(active_vertices_ts);
  // FP: "23 -> 24;
}
__global__ void KCore_TB_LB(CSRGraph graph, unsigned int __begin, unsigned int __end, uint8_t * p_flag, uint8_t * p_pull_flag, uint32_t * p_trim, DynamicBitset& bitset_trim, int * thread_prefix_work_wl, unsigned int num_items, PipeContextT<Worklist2> thread_src_wl)
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
      index_type current_edge;
      src_index = compute_src_and_offset(block_start_src_index, block_end_src_index, current_work+1, thread_prefix_work_wl,num_items, offset);
      src= thread_src_wl.in_wl().dwl[src_index];
      current_edge = (graph).getFirstEdge(src)+ offset;
      {
        index_type dst;
        dst = graph.getAbsDestination(current_edge);
        if (p_pull_flag[dst])
        {
          atomicTestAdd(&p_trim[src], (uint32_t)1);
          bitset_trim.set(src);
        }
      }
      current_work = current_work + nthreads;
    }
    __syncthreads();
  }
  // FP: "45 -> 46;
}
__global__ void KCore(CSRGraph graph, unsigned int __begin, unsigned int __end, uint8_t * p_flag, uint8_t * p_pull_flag, uint32_t * p_trim, DynamicBitset& bitset_trim, PipeContextT<Worklist2> thread_work_wl, PipeContextT<Worklist2> thread_src_wl, bool enable_lb)
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
    // FP: "16 -> 17;
    int threshold = TOTAL_THREADS_1D;
    // FP: "17 -> 18;
    if (pop && (graph).getOutDegree(src) >= threshold)
    {
      index = thread_work_wl.in_wl().push_range(1) ;
      thread_src_wl.in_wl().push_range(1);
      thread_work_wl.in_wl().dwl[index] = (graph).getOutDegree(src);
      thread_src_wl.in_wl().dwl[index] = src;
      pop = false;
    }
    // FP: "20 -> 21;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    // FP: "21 -> 22;
    __shared__ struct { index_type src; } _np_closure [TB_SIZE];
    // FP: "22 -> 23;
    _np_closure[threadIdx.x].src = src;
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
      src = _np_closure[nps.tb.src].src;
      // FP: "51 -> 52;
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
      // FP: "86 -> 87;
      __syncthreads();
      // FP: "87 -> 88;
    }

    // FP: "88 -> 89;
    __syncthreads();
    // FP: "89 -> 90;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "90 -> 91;
    while (_np.work())
    {
      // FP: "91 -> 92;
      int _np_i =0;
      // FP: "92 -> 93;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "93 -> 94;
      __syncthreads();
      // FP: "94 -> 95;

      // FP: "95 -> 96;
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
      // FP: "106 -> 107;
      _np.execute_round_done(ITSIZE);
      // FP: "107 -> 108;
      __syncthreads();
    }
    // FP: "109 -> 110;
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
  }
  // FP: "111 -> 112;
}
__global__ void KCoreSanityCheck(CSRGraph graph, unsigned int __begin, unsigned int __end, uint8_t * p_flag, HGAccumulator<uint64_t> active_vertices)
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
      if (p_flag[src])
      {
        active_vertices.reduce( 1);
      }
    }
  }
  // FP: "11 -> 12;
  active_vertices.thread_exit<cub::BlockReduce<uint64_t, TB_SIZE> >(active_vertices_ts);
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
  cudaDeviceSynchronize();
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
  t_work.init_thread_work(ctx->gg.nnodes);
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  InitializeGraph <<<blocks, threads>>>(ctx->gg, __begin, __end, ctx->current_degree.data.gpu_wr_ptr(), ctx->flag.data.gpu_wr_ptr(), ctx->pull_flag.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr());
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
void LiveUpdate_cuda(unsigned int  __begin, unsigned int  __end, unsigned int & active_vertices, uint32_t local_k_core_num, struct CUDA_Context*  ctx)
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
  LiveUpdate <<<blocks, threads>>>(ctx->gg, __begin, __end, local_k_core_num, ctx->current_degree.data.gpu_wr_ptr(), ctx->flag.data.gpu_wr_ptr(), ctx->pull_flag.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr(), _active_vertices);
  cudaDeviceSynchronize();
  // FP: "9 -> 10;
  check_cuda_kernel;
  // FP: "10 -> 11;
  active_vertices = *(active_verticesval.cpu_rd_ptr());
  // FP: "11 -> 12;
}
void LiveUpdate_allNodes_cuda(unsigned int & active_vertices, uint32_t local_k_core_num, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  LiveUpdate_cuda(0, ctx->gg.nnodes, active_vertices, local_k_core_num, ctx);
  // FP: "2 -> 3;
}
void LiveUpdate_masterNodes_cuda(unsigned int & active_vertices, uint32_t local_k_core_num, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  LiveUpdate_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, active_vertices, local_k_core_num, ctx);
  // FP: "2 -> 3;
}
void LiveUpdate_nodesWithEdges_cuda(unsigned int & active_vertices, uint32_t local_k_core_num, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  LiveUpdate_cuda(0, ctx->numNodesWithEdges, active_vertices, local_k_core_num, ctx);
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
  KCore <<<blocks, __tb_KCore>>>(ctx->gg, __begin, __end, ctx->flag.data.gpu_wr_ptr(), ctx->pull_flag.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr(), *(ctx->trim.is_updated.gpu_rd_ptr()), t_work.thread_work_wl, t_work.thread_src_wl, enable_lb);
  cudaDeviceSynchronize();
  if (enable_lb)
  {
    int num_items = t_work.thread_work_wl.in_wl().nitems();
    if (num_items != 0)
    {
      t_work.compute_prefix_sum();
      cudaDeviceSynchronize();
      KCore_TB_LB <<<blocks, __tb_KCore>>>(ctx->gg, __begin, __end, ctx->flag.data.gpu_wr_ptr(), ctx->pull_flag.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr(), *(ctx->trim.is_updated.gpu_rd_ptr()), t_work.thread_prefix_work_wl.gpu_wr_ptr(), num_items, t_work.thread_src_wl);
      cudaDeviceSynchronize();
      t_work.reset_thread_work();
    }
  }
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
void KCoreSanityCheck_cuda(unsigned int  __begin, unsigned int  __end, uint64_t & active_vertices, struct CUDA_Context*  ctx)
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
  KCoreSanityCheck <<<blocks, threads>>>(ctx->gg, __begin, __end, ctx->flag.data.gpu_wr_ptr(), _active_vertices);
  cudaDeviceSynchronize();
  // FP: "9 -> 10;
  check_cuda_kernel;
  // FP: "10 -> 11;
  active_vertices = *(active_verticesval.cpu_rd_ptr());
  // FP: "11 -> 12;
}
void KCoreSanityCheck_allNodes_cuda(uint64_t & active_vertices, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  KCoreSanityCheck_cuda(0, ctx->gg.nnodes, active_vertices, ctx);
  // FP: "2 -> 3;
}
void KCoreSanityCheck_masterNodes_cuda(uint64_t & active_vertices, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  KCoreSanityCheck_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, active_vertices, ctx);
  // FP: "2 -> 3;
}
void KCoreSanityCheck_nodesWithEdges_cuda(uint64_t & active_vertices, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  KCoreSanityCheck_cuda(0, ctx->numNodesWithEdges, active_vertices, ctx);
  // FP: "2 -> 3;
}
