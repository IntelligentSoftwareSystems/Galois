/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "thread_work.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ tb_lb=False $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=False $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ dyn_lb=False $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
bool enable_lb = false;
#include "bc_level_cuda.cuh"
__global__ void InitializeGraph(CSRGraph graph, unsigned int __begin, unsigned int __end, float * p_betweeness_centrality, float * p_dependency, ShortPathType * p_num_shortest_paths)
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
      p_num_shortest_paths[src]    = 0;
      p_dependency[src]            = 0;
    }
  }
  // FP: "9 -> 10;
}
__global__ void InitializeIteration(CSRGraph graph, unsigned int __begin, unsigned int __end, const uint64_t  local_current_src_node, const uint32_t  local_infinity, uint32_t * p_current_length, float * p_dependency, ShortPathType * p_num_shortest_paths)
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
        p_current_length[src]     = local_infinity;
        p_num_shortest_paths[src] = 0;
      }
      else
      {
        p_current_length[src]     = 0;
        p_num_shortest_paths[src] = 1;
      }
      p_dependency[src]       = 0;
    }
  }
  // FP: "15 -> 16;
}
__global__ void ForwardPass(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t local_r, uint32_t * p_current_length, ShortPathType * p_num_shortest_paths, DynamicBitset& bitset_current_length, DynamicBitset& bitset_num_shortest_paths, HGAccumulator<uint32_t> dga)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  __shared__ cub::BlockReduce<uint32_t, TB_SIZE>::TempStorage dga_ts;
  index_type src_end;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  dga.thread_entry();
  // FP: "3 -> 4;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    index_type current_edge_end;
    bool pop  = src < __end;
    if (pop)
    {
      if (p_current_length[src] == local_r)
      {
      }
      else
      {
        pop = false;
      }
    }
    if (!pop)
    {
      continue;
    }
    current_edge_end = (graph).getFirstEdge((src) + 1);
    for (index_type current_edge = (graph).getFirstEdge(src) + 0; current_edge < current_edge_end; current_edge += 1)
    {
      index_type dst;
      uint32_t new_dist;
      uint32_t old;
      dst = graph.getAbsDestination(current_edge);
      new_dist = 1 + p_current_length[src];
      old = atomicTestMin(&p_current_length[dst], new_dist);
      if (old > new_dist)
      {
        double nsp;
        bitset_current_length.set(dst);
        nsp = p_num_shortest_paths[src];
        atomicTestAdd(&p_num_shortest_paths[dst], nsp);
        bitset_num_shortest_paths.set(dst);
        dga.reduce( 1);
      }
      else
      {
        if (old == new_dist)
        {
          double nsp;
          nsp = p_num_shortest_paths[src];
          atomicTestAdd(&p_num_shortest_paths[dst], nsp);
          bitset_num_shortest_paths.set(dst);
          dga.reduce( 1);
        }
      }
    }
  }
  // FP: "37 -> 38;
  dga.thread_exit<cub::BlockReduce<uint32_t, TB_SIZE> >(dga_ts);
  // FP: "38 -> 39;
}
__global__ void MiddleSync(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t local_infinity, uint32_t * p_current_length, DynamicBitset& bitset_num_shortest_paths)
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
        bitset_num_shortest_paths.set(src);
      }
    }
  }
  // FP: "9 -> 10;
}
__global__ void BackwardPass(CSRGraph graph, unsigned int __begin, unsigned int __end, uint32_t local_r, uint32_t * p_current_length, float * p_dependency, ShortPathType * p_num_shortest_paths, DynamicBitset& bitset_dependency)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  uint32_t dest_to_find;
  index_type src_end;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    index_type current_edge_end;
    bool pop  = src < __end;
    if (pop)
    {
      if (p_current_length[src] == local_r)
      {
        dest_to_find = p_current_length[src] + 1;
      }
      else
      {
        pop = false;
      }
    }
    if (!pop)
    {
      continue;
    }
    current_edge_end = (graph).getFirstEdge((src) + 1);
    for (index_type current_edge = (graph).getFirstEdge(src) + 0; current_edge < current_edge_end; current_edge += 1)
    {
      index_type dst;
      dst = graph.getAbsDestination(current_edge);
      if (dest_to_find == p_current_length[dst])
      {
        float contrib;
        contrib = ((float)1 + p_dependency[dst]) / p_num_shortest_paths[dst];
        p_dependency[src] = p_dependency[src] + contrib;
        bitset_dependency.set(src);
      }
    }
    p_dependency[src] *= p_num_shortest_paths[src];
  }
  // FP: "25 -> 26;
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
        p_betweeness_centrality[src] += p_dependency[src];
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
  DGAccumulator_sum.thread_exit<cub::BlockReduce<float, TB_SIZE> >(DGAccumulator_sum_ts);
  // FP: "16 -> 17;
  DGAccumulator_max.thread_exit<cub::BlockReduce<float, TB_SIZE> >(DGAccumulator_max_ts);
  // FP: "17 -> 18;
  DGAccumulator_min.thread_exit<cub::BlockReduce<float, TB_SIZE> >(DGAccumulator_min_ts);
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
  InitializeGraph <<<blocks, threads>>>(ctx->gg, __begin, __end, ctx->betweeness_centrality.data.gpu_wr_ptr(), ctx->dependency.data.gpu_wr_ptr(), ctx->num_shortest_paths.data.gpu_wr_ptr());
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
void InitializeIteration_cuda(unsigned int  __begin, unsigned int  __end, const uint32_t & local_infinity, const uint64_t & local_current_src_node, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  InitializeIteration <<<blocks, threads>>>(ctx->gg, __begin, __end, local_current_src_node, local_infinity, ctx->current_length.data.gpu_wr_ptr(), ctx->dependency.data.gpu_wr_ptr(), ctx->num_shortest_paths.data.gpu_wr_ptr());
  cudaDeviceSynchronize();
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
void ForwardPass_cuda(unsigned int  __begin, unsigned int  __end, uint32_t & dga, uint32_t local_r, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  HGAccumulator<uint32_t> _dga;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  Shared<uint32_t> dgaval  = Shared<uint32_t>(1);
  // FP: "5 -> 6;
  // FP: "6 -> 7;
  *(dgaval.cpu_wr_ptr()) = 0;
  // FP: "7 -> 8;
  _dga.rv = dgaval.gpu_wr_ptr();
  // FP: "8 -> 9;
  ForwardPass <<<blocks, threads>>>(ctx->gg, __begin, __end, local_r, ctx->current_length.data.gpu_wr_ptr(), ctx->num_shortest_paths.data.gpu_wr_ptr(), *(ctx->current_length.is_updated.gpu_rd_ptr()), *(ctx->num_shortest_paths.is_updated.gpu_rd_ptr()), _dga);
  cudaDeviceSynchronize();
  // FP: "9 -> 10;
  check_cuda_kernel;
  // FP: "10 -> 11;
  dga = *(dgaval.cpu_rd_ptr());
  // FP: "11 -> 12;
}
void ForwardPass_allNodes_cuda(uint32_t & dga, uint32_t local_r, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  ForwardPass_cuda(0, ctx->gg.nnodes, dga, local_r, ctx);
  // FP: "2 -> 3;
}
void ForwardPass_masterNodes_cuda(uint32_t & dga, uint32_t local_r, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  ForwardPass_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, dga, local_r, ctx);
  // FP: "2 -> 3;
}
void ForwardPass_nodesWithEdges_cuda(uint32_t & dga, uint32_t local_r, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  ForwardPass_cuda(0, ctx->numNodesWithEdges, dga, local_r, ctx);
  // FP: "2 -> 3;
}
void MiddleSync_cuda(unsigned int  __begin, unsigned int  __end, const uint32_t local_infinity, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  MiddleSync <<<blocks, threads>>>(ctx->gg, __begin, __end, local_infinity, ctx->current_length.data.gpu_wr_ptr(), *(ctx->num_shortest_paths.is_updated.gpu_rd_ptr()));
  cudaDeviceSynchronize();
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void MiddleSync_allNodes_cuda(const uint32_t local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  MiddleSync_cuda(0, ctx->gg.nnodes, local_infinity, ctx);
  // FP: "2 -> 3;
}
void MiddleSync_masterNodes_cuda(const uint32_t local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  MiddleSync_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_infinity, ctx);
  // FP: "2 -> 3;
}
void MiddleSync_nodesWithEdges_cuda(const uint32_t local_infinity, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  MiddleSync_cuda(0, ctx->numNodesWithEdges, local_infinity, ctx);
  // FP: "2 -> 3;
}
void BackwardPass_cuda(unsigned int  __begin, unsigned int  __end, uint32_t local_r, struct CUDA_Context*  ctx)
{
  dim3 blocks;
  dim3 threads;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  kernel_sizing(blocks, threads);
  // FP: "4 -> 5;
  BackwardPass <<<blocks, threads>>>(ctx->gg, __begin, __end, local_r, ctx->current_length.data.gpu_wr_ptr(), ctx->dependency.data.gpu_wr_ptr(), ctx->num_shortest_paths.data.gpu_wr_ptr(), *(ctx->dependency.is_updated.gpu_rd_ptr()));
  cudaDeviceSynchronize();
  // FP: "5 -> 6;
  check_cuda_kernel;
  // FP: "6 -> 7;
}
void BackwardPass_allNodes_cuda(uint32_t local_r, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  BackwardPass_cuda(0, ctx->gg.nnodes, local_r, ctx);
  // FP: "2 -> 3;
}
void BackwardPass_masterNodes_cuda(uint32_t local_r, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  BackwardPass_cuda(ctx->beginMaster, ctx->beginMaster + ctx->numOwned, local_r, ctx);
  // FP: "2 -> 3;
}
void BackwardPass_nodesWithEdges_cuda(uint32_t local_r, struct CUDA_Context*  ctx)
{
  // FP: "1 -> 2;
  BackwardPass_cuda(0, ctx->numNodesWithEdges, local_r, ctx);
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
  cudaDeviceSynchronize();
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
  *(DGAccumulator_minval.cpu_wr_ptr()) = 1073741823;
  // FP: "15 -> 16;
  _DGAccumulator_min.rv = DGAccumulator_minval.gpu_wr_ptr();
  // FP: "16 -> 17;
  Sanity <<<blocks, threads>>>(ctx->gg, __begin, __end, ctx->betweeness_centrality.data.gpu_wr_ptr(), _DGAccumulator_sum, _DGAccumulator_max, _DGAccumulator_min);
  cudaDeviceSynchronize();
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