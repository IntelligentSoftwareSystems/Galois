/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
unsigned int * P_DIST_CURRENT;
#include "kernels/reduce.cuh"
#include "gen_cuda.cuh"
static const int __tb_SSSP = TB_SIZE;
__global__ void InitializeGraph(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, const unsigned int  local_infinity, unsigned int local_src_node, unsigned int * p_dist_current)
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
__global__ void SSSP(CSRGraph graph, unsigned int __nowned, unsigned int __begin, unsigned int __end, unsigned int * p_dist_current, Sum ret_val)
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
    bool pop  = src < __end;
    if (pop)
    {
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
        index_type jj;
        jj = ns +_np_j;
        {
          index_type dst;
          unsigned int new_dist;
          unsigned int old_dist;
          dst = graph.getAbsDestination(jj);
          new_dist = p_dist_current[dst] + graph.getAbsWeight(jj);
          old_dist = atomicMin(&p_dist_current[src], new_dist);
          if (old_dist > new_dist)
          {
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
          index_type jj;
          jj = _np_w_start +_np_ii;
          {
            index_type dst;
            unsigned int new_dist;
            unsigned int old_dist;
            dst = graph.getAbsDestination(jj);
            new_dist = p_dist_current[dst] + graph.getAbsWeight(jj);
            old_dist = atomicMin(&p_dist_current[src], new_dist);
            if (old_dist > new_dist)
            {
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
        index_type jj;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        src = _np_closure[nps.fg.src[_np_i]].src;
        jj= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          unsigned int new_dist;
          unsigned int old_dist;
          dst = graph.getAbsDestination(jj);
          new_dist = p_dist_current[dst] + graph.getAbsWeight(jj);
          old_dist = atomicMin(&p_dist_current[src], new_dist);
          if (old_dist > new_dist)
          {
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
  }
  ret_val.thread_exit<_br>(_ts);
}
void InitializeGraph_cuda(unsigned int  __begin, unsigned int  __end, const unsigned int & local_infinity, unsigned int local_src_node, struct CUDA_Context * ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  InitializeGraph <<<blocks, threads>>>(ctx->gg, ctx->nowned, __begin, __end, local_infinity, local_src_node, ctx->dist_current.data.gpu_wr_ptr());
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void InitializeGraph_all_cuda(const unsigned int & local_infinity, unsigned int local_src_node, struct CUDA_Context * ctx)
{
  // FP: "1 -> 2;
  InitializeGraph_cuda(0, ctx->nowned, local_infinity, local_src_node, ctx);
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
  SSSP <<<blocks, __tb_SSSP>>>(ctx->gg, ctx->nowned, __begin, __end, ctx->dist_current.data.gpu_wr_ptr(), _rv);
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