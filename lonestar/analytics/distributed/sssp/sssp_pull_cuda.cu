/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "thread_work.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ tb_lb=False $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=False $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ dyn_lb=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
struct ThreadWork t_work;
bool enable_lb = true;
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
__global__ void SSSP_TB_LB(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_dist_current, DynamicBitset& bitset_dist_current, HGAccumulator<unsigned int> active_vertices, int * thread_prefix_work_wl, unsigned int num_items, PipeContextT<Worklist2> thread_src_wl)
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
        new_dist = p_dist_current[dst] + graph.getAbsWeight(jj);
        old_dist = atomicTestMin(&p_dist_current[src], new_dist);
        if (old_dist > new_dist)
        {
          bitset_dist_current.set(src);
          active_vertices.reduce( 1);
        }
      }
      current_work = current_work + nthreads;
    }
    __syncthreads();
  }
  // FP: "50 -> 51;
}
__global__ void SSSP(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t * p_dist_current, DynamicBitset& bitset_dist_current, HGAccumulator<unsigned int> active_vertices, PipeContextT<Worklist2> thread_work_wl, PipeContextT<Worklist2> thread_src_wl, bool enable_lb)
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
    int index;
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    // FP: "8 -> 9;
    bool pop  = src < __end && ((( src < (graph).nnodes )) ? true: false);
    // FP: "9 -> 10;
    if (pop)
    {
    }
    // FP: "11 -> 12;
    // FP: "14 -> 15;
    // FP: "15 -> 16;
    int threshold = TOTAL_THREADS_1D;
    // FP: "16 -> 17;
    if (pop && (graph).getOutDegree(src) >= threshold)
    {
      index = thread_work_wl.in_wl().push_range(1) ;
      thread_src_wl.in_wl().push_range(1);
      thread_work_wl.in_wl().dwl[index] = (graph).getOutDegree(src);
      thread_src_wl.in_wl().dwl[index] = src;
      pop = false;
    }
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
      // FP: "64 -> 65;
      __syncthreads();
    }
    // FP: "66 -> 67;

    // FP: "67 -> 68;
    {
      const int warpid = threadIdx.x / 32;
      // FP: "68 -> 69;
      const int _np_laneid = cub::LaneId();
      // FP: "69 -> 70;
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
      // FP: "93 -> 94;
      __syncthreads();
      // FP: "94 -> 95;
    }

    // FP: "95 -> 96;
    __syncthreads();
    // FP: "96 -> 97;
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    // FP: "97 -> 98;
    while (_np.work())
    {
      // FP: "98 -> 99;
      int _np_i =0;
      // FP: "99 -> 100;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      // FP: "100 -> 101;
      __syncthreads();
      // FP: "101 -> 102;

      // FP: "102 -> 103;
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
      // FP: "117 -> 118;
      _np.execute_round_done(ITSIZE);
      // FP: "118 -> 119;
      __syncthreads();
    }
    // FP: "120 -> 121;
    assert(threadIdx.x < __kernel_tb_size);
    src = _np_closure[threadIdx.x].src;
  }
  // FP: "122 -> 123;
  active_vertices.thread_exit<cub::BlockReduce<unsigned int, TB_SIZE> >(active_vertices_ts);
  // FP: "123 -> 124;
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
  t_work.init_thread_work(ctx->gg.nnodes);
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  InitializeGraph <<<blocks, threads>>>(ctx->gg, __begin, __end, local_infinity, local_src_node, ctx->dist_current.data.gpu_wr_ptr());
  cudaDeviceSynchronize();
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
  SSSP <<<blocks, __tb_SSSP>>>(ctx->gg, __begin, __end, ctx->dist_current.data.gpu_wr_ptr(), *(ctx->dist_current.is_updated.gpu_rd_ptr()), _active_vertices, t_work.thread_work_wl, t_work.thread_src_wl, enable_lb);
  cudaDeviceSynchronize();
  if (enable_lb)
  {
    int num_items = t_work.thread_work_wl.in_wl().nitems();
    if (num_items != 0)
    {
      t_work.compute_prefix_sum();
      cudaDeviceSynchronize();
      SSSP_TB_LB <<<blocks, __tb_SSSP>>>(ctx->gg, __begin, __end, ctx->dist_current.data.gpu_wr_ptr(), *(ctx->dist_current.is_updated.gpu_rd_ptr()), _active_vertices, t_work.thread_prefix_work_wl.gpu_wr_ptr(), num_items, t_work.thread_src_wl);
      cudaDeviceSynchronize();
      t_work.reset_thread_work();
    }
  }
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
  cudaDeviceSynchronize();
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