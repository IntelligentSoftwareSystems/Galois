/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "thread_work.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=True $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'wp']) $ cc_disable=set([]) $ tb_lb=True $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
struct ThreadWork t_work;
extern int DELTA;
extern int start_node;
bool enable_lb = false;
typedef int edge_data_type;
typedef int node_data_type;
typedef int * gint_p;
extern const node_data_type INF = INT_MAX;
static const int __tb_gg_main_pipe_1_gpu_gb = 256;
static const int __tb_sssp_kernel = TB_SIZE;
static const int __tb_remove_dups = TB_SIZE;
__global__ void kernel(CSRGraph graph, int src)
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
__device__ void remove_dups_dev(int * marks, Worklist2 in_wl, Worklist2 out_wl, GlobalBarrier gb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  index_type wlnode_end;
  index_type wlnode2_end;
  wlnode_end = *((volatile index_type *) (in_wl).dindex);
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    int node;
    bool pop;
    pop = (in_wl).pop_id(wlnode, node);
    marks[node] = wlnode;
  }
  gb.Sync();
  wlnode2_end = *((volatile index_type *) (in_wl).dindex);
  for (index_type wlnode2 = 0 + tid; wlnode2 < wlnode2_end; wlnode2 += nthreads)
  {
    int node;
    bool pop;
    pop = (in_wl).pop_id(wlnode2, node);
    if (marks[node] == wlnode2)
    {
      index_type _start_26;
      _start_26 = (out_wl).setup_push_warp_one();;
      (out_wl).do_push(_start_26, 0, node);
    }
  }
}
__global__ void remove_dups(int * marks, Worklist2 in_wl, Worklist2 out_wl, GlobalBarrier gb)
{
  unsigned tid = TID_1D;

  if (tid == 0)
    in_wl.reset_next_slot();

  remove_dups_dev(marks, in_wl, out_wl, gb);
}
__global__ void sssp_kernel_dev_TB_LB(CSRGraph graph, int delta, int * thread_prefix_work_wl, unsigned int num_items, PipeContextT<Worklist2> thread_src_wl, Worklist2 in_wl, Worklist2 out_wl, Worklist2 re_wl)
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
        index_type dst = graph.getAbsDestination(edge);
        edge_data_type wt = graph.getAbsWeight(edge);
        if (graph.node_data[dst] > graph.node_data[node] + wt)
        {
          atomicMin(graph.node_data + dst, graph.node_data[node] + wt);
          if (graph.node_data[node] + wt <= delta)
          {
            index_type _start_67;
            _start_67 = (re_wl).setup_push_warp_one();;
            (re_wl).do_push(_start_67, 0, dst);
          }
          else
          {
            (out_wl).push(dst);
          }
        }
      }
      current_work = current_work + nthreads;
    }
  }
}
__global__ void Inspect_sssp_kernel_dev(CSRGraph graph, int delta, PipeContextT<Worklist2> thread_work_wl, PipeContextT<Worklist2> thread_src_wl, bool enable_lb, Worklist2 in_wl, Worklist2 out_wl)
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
    if (pop && graph.node_data[node] == INF)
    {
      continue;
    }
    if (pop)
    {
      index = thread_work_wl.in_wl().push_range(1) ;
      thread_src_wl.in_wl().push_range(1);
      thread_work_wl.in_wl().dwl[index] = (graph).getOutDegree(node);
      thread_src_wl.in_wl().dwl[index] = node;
    }
  }
}
__device__ void sssp_kernel_dev(CSRGraph graph, int delta, bool enable_lb, Worklist2 in_wl, Worklist2 out_wl, Worklist2 re_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_sssp_kernel;
  index_type wlnode_end;
  const int _NP_CROSSOVER_WP = 32;
  const int BLKSIZE = __kernel_tb_size;
  const int ITSIZE = BLKSIZE * 8;

  typedef cub::BlockScan<multiple_sum<2, index_type>, BLKSIZE> BlockScan;
  typedef union np_shared<BlockScan::TempStorage, index_type, struct empty_np, struct warp_np<__kernel_tb_size/32>, struct fg_np<ITSIZE> > npsTy;

  __shared__ npsTy nps ;
  wlnode_end = roundup((*((volatile index_type *) (in_wl).dindex)), (blockDim.x));
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    int node;
    bool pop;
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    pop = (in_wl).pop_id(wlnode, node) && ((( node < (graph).nnodes ) && ( (graph).getOutDegree(node) < DEGREE_LIMIT)) ? true: false);
    pop = pop && !(graph.node_data[node] == INF);
    struct NPInspector1 _np = {0,0,0,0,0,0};
    __shared__ struct { int node; } _np_closure [TB_SIZE];
    _np_closure[threadIdx.x].node = node;
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
    }
    __syncthreads();
    {
      const int warpid = threadIdx.x / 32;
      const int _np_laneid = cub::LaneId();
      while (__any_sync(0xffffffff, _np.size >= _NP_CROSSOVER_WP))
      {
        if (_np.size >= _NP_CROSSOVER_WP)
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
        node = _np_closure[nps.warp.src[warpid]].node;
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type edge;
          edge = _np_w_start +_np_ii;
          {
            index_type dst = graph.getAbsDestination(edge);
            edge_data_type wt = graph.getAbsWeight(edge);
            if (graph.node_data[dst] > graph.node_data[node] + wt)
            {
              atomicMin(graph.node_data + dst, graph.node_data[node] + wt);
              if (graph.node_data[node] + wt <= delta)
              {
                index_type _start_67;
                _start_67 = (re_wl).setup_push_warp_one();;
                (re_wl).do_push(_start_67, 0, dst);
              }
              else
              {
                (out_wl).push(dst);
              }
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
        index_type edge;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        node = _np_closure[nps.fg.src[_np_i]].node;
        edge= nps.fg.itvalue[_np_i];
        {
          index_type dst = graph.getAbsDestination(edge);
          edge_data_type wt = graph.getAbsWeight(edge);
          if (graph.node_data[dst] > graph.node_data[node] + wt)
          {
            atomicMin(graph.node_data + dst, graph.node_data[node] + wt);
            if (graph.node_data[node] + wt <= delta)
            {
              index_type _start_67;
              _start_67 = (re_wl).setup_push_warp_one();;
              (re_wl).do_push(_start_67, 0, dst);
            }
            else
            {
              (out_wl).push(dst);
            }
          }
        }
      }
      _np.execute_round_done(ITSIZE);
      __syncthreads();
    }
    assert(threadIdx.x < __kernel_tb_size);
    node = _np_closure[threadIdx.x].node;
  }
}
__global__ void __launch_bounds__(TB_SIZE, 2) sssp_kernel(CSRGraph graph, int delta, bool enable_lb, Worklist2 in_wl, Worklist2 out_wl, Worklist2 re_wl)
{
  unsigned tid = TID_1D;

  if (tid == 0)
    in_wl.reset_next_slot();

  sssp_kernel_dev(graph, delta, enable_lb, in_wl, out_wl, re_wl);
}
void gg_main_pipe_1(CSRGraph& gg, gint_p glevel, int& curdelta, int& i, int DELTA, GlobalBarrier& remove_dups_barrier, int remove_dups_blocks, PipeContextT<Worklist2>& pipe, dim3& blocks, dim3& threads)
{
  while (pipe.in_wl().nitems())
  {
    while (pipe.in_wl().nitems())
    {
      pipe.out_wl().will_write();
      pipe.re_wl().will_write();
      if (enable_lb)
      {
        t_work.reset_thread_work();
        Inspect_sssp_kernel_dev <<<blocks, __tb_sssp_kernel>>>(gg, curdelta, t_work.thread_work_wl, t_work.thread_src_wl, enable_lb, pipe.in_wl(), pipe.out_wl());
        cudaDeviceSynchronize();
        int num_items = t_work.thread_work_wl.in_wl().nitems();
        if (num_items != 0)
        {
          t_work.compute_prefix_sum();
          cudaDeviceSynchronize();
          sssp_kernel_dev_TB_LB <<<blocks, __tb_sssp_kernel>>>(gg, curdelta, t_work.thread_prefix_work_wl.gpu_wr_ptr(), num_items, t_work.thread_src_wl, pipe.in_wl(), pipe.out_wl(), pipe.re_wl());
          cudaDeviceSynchronize();
        }
      }
      sssp_kernel <<<blocks, __tb_sssp_kernel>>>(gg, curdelta, enable_lb, pipe.in_wl(), pipe.out_wl(), pipe.re_wl());
      cudaDeviceSynchronize();
      pipe.in_wl().swap_slots();
      pipe.retry2();
    }
    pipe.advance2();
    pipe.out_wl().will_write();
    remove_dups <<<remove_dups_blocks, __tb_remove_dups>>>(glevel, pipe.in_wl(), pipe.out_wl(), remove_dups_barrier);
    cudaDeviceSynchronize();
    pipe.in_wl().swap_slots();
    pipe.advance2();
    i++;
    curdelta += DELTA;
  }
}
__global__ void __launch_bounds__(__tb_gg_main_pipe_1_gpu_gb) gg_main_pipe_1_gpu_gb(CSRGraph gg, gint_p glevel, int curdelta, int i, int DELTA, GlobalBarrier remove_dups_barrier, int remove_dups_blocks, PipeContextT<Worklist2> pipe, int* cl_curdelta, int* cl_i, bool enable_lb, GlobalBarrier gb)
{
  unsigned tid = TID_1D;

  curdelta = *cl_curdelta;
  i = *cl_i;
  while (pipe.in_wl().nitems())
  {
    while (pipe.in_wl().nitems())
    {
      if (tid == 0)
        pipe.in_wl().reset_next_slot();
      sssp_kernel_dev (gg, curdelta, enable_lb, pipe.in_wl(), pipe.out_wl(), pipe.re_wl());
      pipe.in_wl().swap_slots();
      gb.Sync();
      pipe.retry2();
    }
    gb.Sync();
    pipe.advance2();
    if (tid == 0)
      pipe.in_wl().reset_next_slot();
    remove_dups_dev (glevel, pipe.in_wl(), pipe.out_wl(), gb);
    pipe.in_wl().swap_slots();
    gb.Sync();
    pipe.advance2();
    i++;
    curdelta += DELTA;
  }
  gb.Sync();
  if (tid == 0)
  {
    *cl_curdelta = curdelta;
    *cl_i = i;
  }
}
__global__ void gg_main_pipe_1_gpu(CSRGraph gg, gint_p glevel, int curdelta, int i, int DELTA, GlobalBarrier remove_dups_barrier, int remove_dups_blocks, PipeContextT<Worklist2> pipe, dim3 blocks, dim3 threads, int* cl_curdelta, int* cl_i, bool enable_lb)
{
  unsigned tid = TID_1D;

  curdelta = *cl_curdelta;
  i = *cl_i;
  while (pipe.in_wl().nitems())
  {
    while (pipe.in_wl().nitems())
    {
      sssp_kernel <<<blocks, __tb_sssp_kernel>>>(gg, curdelta, enable_lb, pipe.in_wl(), pipe.out_wl(), pipe.re_wl());
      cudaDeviceSynchronize();
      pipe.in_wl().swap_slots();
      cudaDeviceSynchronize();
      pipe.retry2();
    }
    cudaDeviceSynchronize();
    pipe.advance2();
    remove_dups <<<remove_dups_blocks, __tb_remove_dups>>>(glevel, pipe.in_wl(), pipe.out_wl(), remove_dups_barrier);
    cudaDeviceSynchronize();
    pipe.in_wl().swap_slots();
    cudaDeviceSynchronize();
    pipe.advance2();
    i++;
    curdelta += DELTA;
  }
  if (tid == 0)
  {
    *cl_curdelta = curdelta;
    *cl_i = i;
  }
}
void gg_main_pipe_1_wrapper(CSRGraph& gg, gint_p glevel, int& curdelta, int& i, int DELTA, GlobalBarrier& remove_dups_barrier, int remove_dups_blocks, PipeContextT<Worklist2>& pipe, dim3& blocks, dim3& threads)
{
  static GlobalBarrierLifetime gg_main_pipe_1_gpu_gb_barrier;
  static bool gg_main_pipe_1_gpu_gb_barrier_inited;
  extern bool enable_lb;
  static const size_t gg_main_pipe_1_gpu_gb_residency = maximum_residency(gg_main_pipe_1_gpu_gb, __tb_gg_main_pipe_1_gpu_gb, 0);
  static const size_t gg_main_pipe_1_gpu_gb_blocks = GG_MIN(blocks.x, ggc_get_nSM() * gg_main_pipe_1_gpu_gb_residency);
  if(!gg_main_pipe_1_gpu_gb_barrier_inited) { gg_main_pipe_1_gpu_gb_barrier.Setup(gg_main_pipe_1_gpu_gb_blocks); gg_main_pipe_1_gpu_gb_barrier_inited = true;};
  if (enable_lb)
  {
    gg_main_pipe_1(gg,glevel,curdelta,i,DELTA,remove_dups_barrier,remove_dups_blocks,pipe,blocks,threads);
  }
  else
  {
    int* cl_curdelta;
    int* cl_i;
    check_cuda(cudaMalloc(&cl_curdelta, sizeof(int) * 1));
    check_cuda(cudaMalloc(&cl_i, sizeof(int) * 1));
    check_cuda(cudaMemcpy(cl_curdelta, &curdelta, sizeof(int) * 1, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(cl_i, &i, sizeof(int) * 1, cudaMemcpyHostToDevice));

    // gg_main_pipe_1_gpu<<<1,1>>>(gg,glevel,curdelta,i,DELTA,remove_dups_barrier,remove_dups_blocks,pipe,blocks,threads,cl_curdelta,cl_i, enable_lb);
    gg_main_pipe_1_gpu_gb<<<gg_main_pipe_1_gpu_gb_blocks, __tb_gg_main_pipe_1_gpu_gb>>>(gg,glevel,curdelta,i,DELTA,remove_dups_barrier,remove_dups_blocks,pipe,cl_curdelta,cl_i, enable_lb, gg_main_pipe_1_gpu_gb_barrier);
    check_cuda(cudaMemcpy(&curdelta, cl_curdelta, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(&i, cl_i, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    check_cuda(cudaFree(cl_curdelta));
    check_cuda(cudaFree(cl_i));
  }
}
void gg_main(CSRGraph& hg, CSRGraph& gg)
{
  dim3 blocks, threads;
  kernel_sizing(gg, blocks, threads);
  t_work.init_thread_work(gg.nnodes);
  static GlobalBarrierLifetime remove_dups_barrier;
  static bool remove_dups_barrier_inited;
  gint_p glevel;
  PipeContextT<Worklist2> pipe;
  Shared<int> level (hg.nnodes);
  level.cpu_wr_ptr();
  static const size_t remove_dups_residency = maximum_residency(remove_dups, __tb_remove_dups, 0);
  static const size_t remove_dups_blocks = GG_MIN(blocks.x, ggc_get_nSM() * remove_dups_residency);
  if(!remove_dups_barrier_inited) { remove_dups_barrier.Setup(remove_dups_blocks); remove_dups_barrier_inited = true;};
  kernel <<<blocks, threads>>>(gg, start_node);
  cudaDeviceSynchronize();
  int i = 0;
  int curdelta = 0;
  printf("delta: %d\n", DELTA);
  glevel = level.gpu_wr_ptr();
  pipe = PipeContextT<Worklist2>(gg.nedges*2);
  pipe.in_wl().wl[0] = start_node;
  pipe.in_wl().update_gpu(1);
  gg_main_pipe_1_wrapper(gg,glevel,curdelta,i,DELTA,remove_dups_barrier,remove_dups_blocks,pipe,blocks,threads);
  printf("iterations: %d\n", i);
}
