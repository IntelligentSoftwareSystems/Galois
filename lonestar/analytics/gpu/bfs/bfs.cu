/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "thread_work.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=True $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ tb_lb=True $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
struct ThreadWork t_work;
extern int start_node;
bool enable_lb = false;
typedef int edge_data_type;
typedef int node_data_type;
extern const node_data_type INF = INT_MAX;
static const int __tb_bfs_kernel = TB_SIZE;
static const int __tb_gg_main_pipe_1_gpu_gb = 256;
__global__ void bfs_init(CSRGraph graph, int src)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  index_type node_end;
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    graph.node_data[node] = (node == src) ? 0 : INF ;
  }
}
__global__ void bfs_kernel_dev_TB_LB(CSRGraph graph, int LEVEL, int * thread_prefix_work_wl, unsigned int num_items, PipeContextT<Worklist2> thread_src_wl, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  __shared__ unsigned int total_work;
  __shared__ unsigned block_start_src_index;
  __shared__ unsigned block_end_src_index;
  unsigned my_work;
  unsigned node;
  unsigned int offset;
  unsigned int current_work;
  unsigned blockdim_x = BLOCK_DIM_X;
  total_work = thread_prefix_work_wl[num_items - 1];
  my_work = ceilf((float)(total_work) / (float) nthreads);

  __syncthreads();

  if (my_work != 0)
  {
    current_work = tid;
  }
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
      index_type edge;
      src_index = compute_src_and_offset(block_start_src_index, block_end_src_index, current_work+1, thread_prefix_work_wl,num_items, offset);
      node= thread_src_wl.in_wl().dwl[src_index];
      edge = (graph).getFirstEdge(node)+ offset;
      {
        index_type dst;
        dst = graph.getAbsDestination(edge);
        if (graph.node_data[dst] == INF)
        {
          index_type _start_24;
          graph.node_data[dst] = LEVEL;
          _start_24 = (out_wl).setup_push_warp_one();;
          (out_wl).do_push(_start_24, 0, dst);
        }
      }
      current_work = current_work + nthreads;
    }
  }
}
__global__ void Inspect_bfs_kernel_dev(CSRGraph graph, int LEVEL, PipeContextT<Worklist2> thread_work_wl, PipeContextT<Worklist2> thread_src_wl, bool enable_lb, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  index_type wlnode_end;
  wlnode_end = *((volatile index_type *) (in_wl).dindex);
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    int node;
    bool pop;
    int index;
    pop = (in_wl).pop_id(wlnode, node) && ((( node < (graph).nnodes ) && ( (graph).getOutDegree(node) >= DEGREE_LIMIT)) ? true: false);
    if (pop)
    {
      index = thread_work_wl.in_wl().push_range(1) ;
      thread_src_wl.in_wl().push_range(1);
      thread_work_wl.in_wl().dwl[index] = (graph).getOutDegree(node);
      thread_src_wl.in_wl().dwl[index] = node;
    }
  }
}
__device__ void bfs_kernel_dev(CSRGraph graph, int LEVEL, bool enable_lb, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_bfs_kernel;
  index_type wlnode_end;
  const int _NP_CROSSOVER_WP = 32;
  const int _NP_CROSSOVER_TB = __kernel_tb_size;
  const int BLKSIZE = __kernel_tb_size;
  const int ITSIZE = BLKSIZE * 8;

  typedef cub::BlockScan<multiple_sum<2, index_type>, BLKSIZE> BlockScan;
  typedef union np_shared<BlockScan::TempStorage, index_type, struct tb_np, struct warp_np<__kernel_tb_size/32>, struct fg_np<ITSIZE> > npsTy;

  __shared__ npsTy nps ;
  wlnode_end = roundup((*((volatile index_type *) (in_wl).dindex)), (blockDim.x));
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    int node;
    bool pop;
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    pop = (in_wl).pop_id(wlnode, node) && ((( node < (graph).nnodes ) && ( (graph).getOutDegree(node) < DEGREE_LIMIT)) ? true: false);
    struct NPInspector1 _np = {0,0,0,0,0,0};
    if (pop)
    {
      _np.size = (graph).getOutDegree(node);
      _np.start = (graph).getFirstEdge(node);
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
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type edge;
        edge = ns +_np_j;
        {
          index_type dst;
          dst = graph.getAbsDestination(edge);
          if (graph.node_data[dst] == INF)
          {
            index_type _start_24;
            graph.node_data[dst] = LEVEL;
            _start_24 = (out_wl).setup_push_warp_one();;
            (out_wl).do_push(_start_24, 0, dst);
          }
        }
      }
      __syncthreads();
    }

    {
      const int warpid = threadIdx.x / 32;
      const int _np_laneid = cub::LaneId();
      while (__any_sync(0xffffffff,_np.size >= _NP_CROSSOVER_WP && _np.size < _NP_CROSSOVER_TB))
      {
        if (_np.size >= _NP_CROSSOVER_WP && _np.size < _NP_CROSSOVER_TB)
        {
          nps.warp.owner[warpid] = _np_laneid;
        }
        if (nps.warp.owner[warpid] == _np_laneid)
        {
          nps.warp.start[warpid] = _np.start;
          nps.warp.size[warpid] = _np.size;

          _np.start = 0;
          _np.size = 0;
        }
        index_type _np_w_start = nps.warp.start[warpid];
        index_type _np_w_size = nps.warp.size[warpid];
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type edge;
          edge = _np_w_start +_np_ii;
          {
            index_type dst;
            dst = graph.getAbsDestination(edge);
            if (graph.node_data[dst] == INF)
            {
              index_type _start_24;
              graph.node_data[dst] = LEVEL;
              _start_24 = (out_wl).setup_push_warp_one();;
              (out_wl).do_push(_start_24, 0, dst);
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
      _np.inspect(nps.fg.itvalue, ITSIZE);
      __syncthreads();

      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type edge;
        edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          dst = graph.getAbsDestination(edge);
          if (graph.node_data[dst] == INF)
          {
            index_type _start_24;
            graph.node_data[dst] = LEVEL;
            _start_24 = (out_wl).setup_push_warp_one();;
            (out_wl).do_push(_start_24, 0, dst);
          }
        }
      }
      _np.execute_round_done(ITSIZE);
      __syncthreads();
    }
  }
}
__global__ void bfs_kernel(CSRGraph graph, int LEVEL, bool enable_lb, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;

  if (tid == 0)
    in_wl.reset_next_slot();

  bfs_kernel_dev(graph, LEVEL, enable_lb, in_wl, out_wl);
}
void gg_main_pipe_1(CSRGraph& gg, int& LEVEL, PipeContextT<Worklist2>& pipe, dim3& blocks, dim3& threads)
{
  while (pipe.in_wl().nitems())
  {
    pipe.out_wl().will_write();
    if (enable_lb)
    {
      t_work.reset_thread_work();
      Inspect_bfs_kernel_dev <<<blocks, __tb_bfs_kernel>>>(gg, LEVEL, t_work.thread_work_wl, t_work.thread_src_wl, enable_lb, pipe.in_wl(), pipe.out_wl());
      cudaDeviceSynchronize();
      int num_items = t_work.thread_work_wl.in_wl().nitems();
      if (num_items != 0)
      {
        t_work.compute_prefix_sum();
        cudaDeviceSynchronize();
        bfs_kernel_dev_TB_LB <<<blocks, __tb_bfs_kernel>>>(gg, LEVEL, t_work.thread_prefix_work_wl.gpu_wr_ptr(), num_items, t_work.thread_src_wl, pipe.in_wl(), pipe.out_wl());
        cudaDeviceSynchronize();
      }
    }
    bfs_kernel <<<blocks, __tb_bfs_kernel>>>(gg, LEVEL, enable_lb, pipe.in_wl(), pipe.out_wl());
    cudaDeviceSynchronize();
    pipe.in_wl().swap_slots();
    pipe.advance2();
    LEVEL++;
  }
}
__global__ void __launch_bounds__(__tb_gg_main_pipe_1_gpu_gb) gg_main_pipe_1_gpu_gb(CSRGraph gg, int LEVEL, PipeContextT<Worklist2> pipe, int* cl_LEVEL, bool enable_lb, GlobalBarrier gb)
{
  unsigned tid = TID_1D;

  LEVEL = *cl_LEVEL;
  while (pipe.in_wl().nitems())
  {
    if (tid == 0)
      pipe.in_wl().reset_next_slot();
    bfs_kernel_dev (gg, LEVEL, enable_lb, pipe.in_wl(), pipe.out_wl());
    pipe.in_wl().swap_slots();
    gb.Sync();
    pipe.advance2();
    LEVEL++;
  }
  gb.Sync();
  if (tid == 0)
  {
    *cl_LEVEL = LEVEL;
  }
}
__global__ void gg_main_pipe_1_gpu(CSRGraph gg, int LEVEL, PipeContextT<Worklist2> pipe, dim3 blocks, dim3 threads, int* cl_LEVEL, bool enable_lb)
{
  unsigned tid = TID_1D;

  LEVEL = *cl_LEVEL;
  while (pipe.in_wl().nitems())
  {
    bfs_kernel <<<blocks, __tb_bfs_kernel>>>(gg, LEVEL, enable_lb, pipe.in_wl(), pipe.out_wl());
    cudaDeviceSynchronize();
    pipe.in_wl().swap_slots();
    cudaDeviceSynchronize();
    pipe.advance2();
    LEVEL++;
  }
  if (tid == 0)
  {
    *cl_LEVEL = LEVEL;
  }
}
void gg_main_pipe_1_wrapper(CSRGraph& gg, int& LEVEL, PipeContextT<Worklist2>& pipe, dim3& blocks, dim3& threads)
{
  static GlobalBarrierLifetime gg_main_pipe_1_gpu_gb_barrier;
  static bool gg_main_pipe_1_gpu_gb_barrier_inited;
  extern bool enable_lb;
  static const size_t gg_main_pipe_1_gpu_gb_residency = maximum_residency(gg_main_pipe_1_gpu_gb, __tb_gg_main_pipe_1_gpu_gb, 0);
  static const size_t gg_main_pipe_1_gpu_gb_blocks = GG_MIN(blocks.x, ggc_get_nSM() * gg_main_pipe_1_gpu_gb_residency);
  if(!gg_main_pipe_1_gpu_gb_barrier_inited) { gg_main_pipe_1_gpu_gb_barrier.Setup(gg_main_pipe_1_gpu_gb_blocks); gg_main_pipe_1_gpu_gb_barrier_inited = true;};
  if (enable_lb)
  {
    gg_main_pipe_1(gg,LEVEL,pipe,blocks,threads);
  }
  else
  {
    int* cl_LEVEL;
    check_cuda(cudaMalloc(&cl_LEVEL, sizeof(int) * 1));
    check_cuda(cudaMemcpy(cl_LEVEL, &LEVEL, sizeof(int) * 1, cudaMemcpyHostToDevice));

    // gg_main_pipe_1_gpu<<<1,1>>>(gg,LEVEL,pipe,blocks,threads,cl_LEVEL, enable_lb);
    gg_main_pipe_1_gpu_gb<<<gg_main_pipe_1_gpu_gb_blocks, __tb_gg_main_pipe_1_gpu_gb>>>(gg,LEVEL,pipe,cl_LEVEL, enable_lb, gg_main_pipe_1_gpu_gb_barrier);
    check_cuda(cudaMemcpy(&LEVEL, cl_LEVEL, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    check_cuda(cudaFree(cl_LEVEL));
  }
}
void gg_main(CSRGraph& hg, CSRGraph& gg)
{
  dim3 blocks, threads;
  kernel_sizing(gg, blocks, threads);
  t_work.init_thread_work(gg.nnodes);
  PipeContextT<Worklist2> wl;
  bfs_init <<<blocks, threads>>>(gg, start_node);
  cudaDeviceSynchronize();
  int LEVEL = 1;
  wl = PipeContextT<Worklist2>(gg.nnodes);
  wl.in_wl().wl[0] = start_node;
  wl.in_wl().update_gpu(1);
  gg_main_pipe_1_wrapper(gg,LEVEL,wl,blocks,threads);
}
