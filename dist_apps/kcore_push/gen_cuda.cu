/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
unsigned int * P_CURRENT_DEGREE;
uint8_t * P_FLAG;
unsigned int * P_TRIM;
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
static const int __tb_KCoreStep1 = TB_SIZE;
static const int __tb_InitializeGraph2 = TB_SIZE;
__global__ void InitializeGraph2(CSRGraph graph, 
    DynamicBitset* is_updated,
    unsigned int __nowned, unsigned int __begin, unsigned int __end, uint32_t * p_current_degree)
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
          atomicAdd(&p_current_degree[dest_node], (uint32_t)1);
          is_updated->set(dest_node);
        }
      }
      // FP: "50 -> 51;
      __syncthreads();
    }
    // FP: "52 -> 53;

    // FP: "53 -> 54;
    {
      const int warpid = threadIdx.x / 32;
      // FP: "54 -> 55;
      const int _np_laneid = cub::LaneId();
      // FP: "55 -> 56;
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
            atomicAdd(&p_current_degree[dest_node], (uint32_t)1);
            is_updated->set(dest_node);

          }
        }
      }
      // FP: "72 -> 73;
      __syncthreads();
      // FP: "73 -> 74;
    }

    // FP: "74 -> 75;
    __syncthreads();
    // FP: "75 -> 76;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "76 -> 77;
    while (_np.work())
    {
      // FP: "77 -> 78;
      int _np_i =0;
      // FP: "78 -> 79;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "79 -> 80;
      __syncthreads();
      // FP: "80 -> 81;

      // FP: "81 -> 82;
      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type current_edge;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        current_edge= nps.fg.itvalue[_np_i];
        {
          index_type dest_node;
          dest_node = graph.getAbsDestination(current_edge);
          atomicAdd(&p_current_degree[dest_node], (uint32_t)1);
          is_updated->set(dest_node);
        }
      }
      // FP: "89 -> 90;
      _np.execute_round_done(ITSIZE);
      // FP: "90 -> 91;
      __syncthreads();
    }
    // FP: "92 -> 93;
    assert(threadIdx.x < __kernel_tb_size);
  }
  // FP: "94 -> 95;
}
__global__ void InitializeGraph1(CSRGraph graph, 
                                 unsigned int __nowned, 
                                 unsigned int __begin, 
                                 unsigned int __end, 
                                 uint32_t * p_current_degree, 
                                 uint8_t * p_flag, 
                                 uint32_t * p_trim)
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
__global__ void KCoreStep2(CSRGraph graph,
  unsigned int __nowned, unsigned int __begin, unsigned int __end, uint32_t * p_current_degree, uint32_t * p_trim, uint8_t * p_flag)
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
      // manual addition to match current cpu code; gpu should have been able
      // to generate it
      if (p_flag[src]) 
      {
        if (p_trim[src] > 0)
        {
          p_current_degree[src] = p_current_degree[src] - p_trim[src];
          p_trim[src] = 0;
        }
      }
    }
  }
  // FP: "10 -> 11;
}
__global__ void KCoreStep1(CSRGraph graph, DynamicBitset* is_updated,
    unsigned int __nowned, unsigned int __begin, unsigned int __end, uint32_t local_k_core_num, uint32_t * p_current_degree, uint8_t * p_flag, uint32_t * p_trim, Sum ret_val)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_KCoreStep1;
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
      if (p_flag[src])
      {
        if (p_current_degree[src] < local_k_core_num)
        {
          p_flag[src] = false;
          ret_val.do_return( 1);
          //continue;
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
    struct NPInspector1 _np = {0,0,0,0,0,0};
    __shared__ struct { ; } _np_closure [TB_SIZE];
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
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type current_edge;
        current_edge = ns +_np_j;
        {
          index_type dst;
          dst = graph.getAbsDestination(current_edge);
          atomicAdd(&p_trim[dst], (uint32_t)1);
          is_updated->set(dst);
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
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type current_edge;
          current_edge = _np_w_start +_np_ii;
          {
            index_type dst;
            dst = graph.getAbsDestination(current_edge);
            atomicAdd(&p_trim[dst], (uint32_t)1);
            is_updated->set(dst);
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
        current_edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          dst = graph.getAbsDestination(current_edge);
          atomicAdd(&p_trim[dst], (uint32_t)1);
          is_updated->set(dst);
        }
      }
      _np.execute_round_done(ITSIZE);
      __syncthreads();
    }
    assert(threadIdx.x < __kernel_tb_size);
  }
  ret_val.thread_exit<_br>(_ts);
}
void InitializeGraph2_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  InitializeGraph2 <<<blocks, __tb_InitializeGraph2>>>(ctx->gg, 
  ctx->current_degree.is_updated.gpu_rd_ptr(),
  ctx->nowned, __begin, __end, ctx->current_degree.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void InitializeGraph2_all_cuda(struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  InitializeGraph2_cuda(0, ctx->nowned, ctx);
  // FP: "2 -> 3;
}
void InitializeGraph1_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  InitializeGraph1 <<<blocks, threads>>>(ctx->gg, 
    ctx->nowned, __begin, __end, ctx->current_degree.data.gpu_wr_ptr(), 
    ctx->flag.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void InitializeGraph1_all_cuda(struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  InitializeGraph1_cuda(0, ctx->nowned, ctx);
  // FP: "2 -> 3;
}
void KCoreStep2_cuda(unsigned int  __begin, unsigned int  __end, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  KCoreStep2 <<<blocks, threads>>>(ctx->gg, 
    ctx->nowned, __begin, __end, ctx->current_degree.data.gpu_wr_ptr(), 
    ctx->trim.data.gpu_wr_ptr(), ctx->flag.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void KCoreStep2_all_cuda(struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  KCoreStep2_cuda(0, ctx->nowned, ctx);
  // FP: "2 -> 3;
}
void KCoreStep1_cuda(unsigned int  __begin, unsigned int  __end, int & __retval, uint32_t local_k_core_num, struct CUDA_Context * ctx)
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
  KCoreStep1 <<<blocks, __tb_KCoreStep1>>>(ctx->gg, 
  ctx->trim.is_updated.gpu_rd_ptr(),
  ctx->nowned, __begin, __end, local_k_core_num, ctx->current_degree.data.gpu_wr_ptr(), ctx->flag.data.gpu_wr_ptr(), ctx->trim.data.gpu_wr_ptr(), _rv);
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
  __retval = *(retval.cpu_rd_ptr());
  // FP: "7 -> 8;
}
void KCoreStep1_all_cuda(int & __retval, uint32_t local_k_core_num, struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  KCoreStep1_cuda(0, ctx->nowned, __retval, local_k_core_num, ctx);
  // FP: "2 -> 3;
}
