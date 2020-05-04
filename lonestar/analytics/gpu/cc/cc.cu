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
extern unsigned long DISCOUNT_TIME_NS;
bool enable_lb = true;
static const int __tb_one = 1;
static const int __tb_prep_edge_src = TB_SIZE;
static const int __tb_gg_main_pipe_4_gpu_gb = 256;
static const int __tb_gg_main_pipe_3_gpu_gb = 256;
__global__ void init(CSRGraph graph)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type node_end;
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    graph.node_data[node] = node;
  }
}
__global__ void prep_edge_src_TB_LB(CSRGraph graph, index_type * edge_src, int * thread_prefix_work_wl, unsigned int num_items, PipeContextT<Worklist2> thread_src_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
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
        edge_src[edge] = node;
      }
      current_work = current_work + nthreads;
    }
  }
}
__global__ void Inspect_prep_edge_src(CSRGraph graph, index_type * edge_src, PipeContextT<Worklist2> thread_work_wl, PipeContextT<Worklist2> thread_src_wl, bool enable_lb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type node_end;
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    bool pop;
    int index;
    pop = (((( node < (graph).nnodes ) && ( (graph).getOutDegree(node) >= DEGREE_LIMIT)) ? true: false)) && graph.valid_node(node);;
    if (pop)
    {
      index = thread_work_wl.in_wl().push_range(1) ;
      thread_src_wl.in_wl().push_range(1);
      thread_work_wl.in_wl().dwl[index] = (graph).getOutDegree(node);
      thread_src_wl.in_wl().dwl[index] = node;
    }
  }
}
__global__ void prep_edge_src(CSRGraph graph, index_type * edge_src, bool enable_lb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_prep_edge_src;
  index_type node_end;
  const int _NP_CROSSOVER_WP = 32;
  const int _NP_CROSSOVER_TB = __kernel_tb_size;
  const int BLKSIZE = __kernel_tb_size;
  const int ITSIZE = BLKSIZE * 8;
  unsigned d_limit = DEGREE_LIMIT;

  typedef cub::BlockScan<multiple_sum<2, index_type>, BLKSIZE> BlockScan;
  typedef union np_shared<BlockScan::TempStorage, index_type, struct tb_np, struct warp_np<__kernel_tb_size/32>, struct fg_np<ITSIZE> > npsTy;

  __shared__ npsTy nps ;
  node_end = roundup(((graph).nnodes), (blockDim.x));
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    bool pop;
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    pop = (((( node < (graph).nnodes ) && ( (graph).getOutDegree(node) < DEGREE_LIMIT)) ? true: false)) && graph.valid_node(node);;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    __shared__ struct { index_type node; } _np_closure [TB_SIZE];
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
      node = _np_closure[nps.tb.src].node;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type edge;
        edge = ns +_np_j;
        {
          edge_src[edge] = node;
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
        node = _np_closure[nps.warp.src[warpid]].node;
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type edge;
          edge = _np_w_start +_np_ii;
          {
            edge_src[edge] = node;
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
          edge_src[edge] = node;
        }
      }
      _np.execute_round_done(ITSIZE);
      __syncthreads();
    }
    assert(threadIdx.x < __kernel_tb_size);
    node = _np_closure[threadIdx.x].node;
  }
}
__global__ void hook_init(CSRGraph graph, index_type * edge_src)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  int edge_end;
  edge_end = graph.nedges;
  for (int edge = 0 + tid; edge < edge_end; edge += nthreads)
  {
    index_type x = edge_src[edge];
    index_type y = graph.getAbsDestination(edge);
    index_type mx = x > y ? x : y;
    index_type mn = x > y ? y : x;
    graph.node_data[mx] = mn;
  }
}
__global__ void hook_high_to_low(CSRGraph graph, const __restrict__ index_type * edge_src, char * marks, HGAccumulator<int> ret_val)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  typedef cub::BlockReduce<int, TB_SIZE> _br;
  __shared__ _br::TempStorage _ts;
  ret_val.thread_entry();
  int edge_end;
  edge_end = graph.nedges;
  for (int edge = 0 + tid; edge < edge_end; edge += nthreads)
  {
    if (!marks[edge])
    {
      index_type u = edge_src[edge];
      index_type v = graph.getAbsDestination(edge);
      node_data_type p_u = graph.node_data[u];
      node_data_type p_v = graph.node_data[v];
      index_type mx = p_u > p_v ? p_u : p_v;
      index_type mn = p_u > p_v ? p_v : p_u;
      if (mx == mn)
      {
        marks[edge] = 1;
      }
      else
      {
        graph.node_data[mn] = mx;
        ret_val.reduce(true);
        continue;
        continue;
      }
    }
  }
  ret_val.thread_exit<_br>(_ts);
}
__global__ void hook_low_to_high(CSRGraph graph, index_type * edge_src, char * marks, HGAccumulator<int> ret_val)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  typedef cub::BlockReduce<int, TB_SIZE> _br;
  __shared__ _br::TempStorage _ts;
  ret_val.thread_entry();
  int edge_end;
  edge_end = graph.nedges;
  for (int edge = 0 + tid; edge < edge_end; edge += nthreads)
  {
    if (!marks[edge])
    {
      index_type u = edge_src[edge];
      index_type v = graph.getAbsDestination(edge);
      node_data_type p_u = graph.node_data[u];
      node_data_type p_v = graph.node_data[v];
      index_type mx = p_u > p_v ? p_u : p_v;
      index_type mn = p_u > p_v ? p_v : p_u;
      if (mx == mn)
      {
        marks[edge] = 1;
      }
      else
      {
        graph.node_data[mx] = mn;
        ret_val.reduce(true);
        continue;
        continue;
      }
    }
  }
  ret_val.thread_exit<_br>(_ts);
}
__device__ void p_jump_dev(CSRGraph graph, HGAccumulator<int> ret_val)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  typedef cub::BlockReduce<int, TB_SIZE> _br;
  __shared__ _br::TempStorage _ts;
  ret_val.thread_entry();
  index_type node_end;
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    node_data_type p_u = graph.node_data[node];
    node_data_type p_v = graph.node_data[p_u];
    if (p_u != p_v)
    {
      graph.node_data[node] = p_v;
      ret_val.reduce(true);
      continue;
      continue;
    }
  }
  ret_val.thread_exit<_br>(_ts);
}
__global__ void p_jump(CSRGraph graph, HGAccumulator<int> ret_val)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  p_jump_dev(graph, ret_val);
}
__global__ void identify_roots(CSRGraph graph, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type node_end;
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    if (graph.node_data[node] == node)
    {
      index_type _start_73;
      _start_73 = (out_wl).setup_push_warp_one();;
      (out_wl).do_push(_start_73, 0, node);
    }
  }
}
__device__ void p_jump_roots_dev(CSRGraph graph, Worklist2 in_wl, Worklist2 out_wl, HGAccumulator<int> ret_val)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  typedef cub::BlockReduce<int, TB_SIZE> _br;
  __shared__ _br::TempStorage _ts;
  ret_val.thread_entry();
  index_type wlnode_end;
  wlnode_end = *((volatile index_type *) (in_wl).dindex);
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    bool pop;
    int node;
    pop = (in_wl).pop_id(wlnode, node);
    node_data_type p_u = graph.node_data[node];
    node_data_type p_v = graph.node_data[p_u];
    if (p_u != p_v)
    {
      graph.node_data[node] = p_v;
      ret_val.reduce(true);
      continue;
      continue;
    }
  }
  ret_val.thread_exit<_br>(_ts);
}
__global__ void p_jump_roots(CSRGraph graph, Worklist2 in_wl, Worklist2 out_wl, HGAccumulator<int> ret_val)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  p_jump_roots_dev(graph, in_wl, out_wl, ret_val);
}
__global__ void p_jump_leaves(CSRGraph graph)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type node_end;
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    node_data_type p_u = graph.node_data[node];
    node_data_type p_v = graph.node_data[p_u];
    if (p_u != p_v)
    {
      graph.node_data[node] = p_v;
    }
  }
}
__global__ void count_components(CSRGraph graph, int * count)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type node_end;
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    if (node == graph.node_data[node])
    {
      atomicAdd(count, 1);
    }
  }
}
void gg_main_pipe_4(CSRGraphTy& gg, PipeContextT<Worklist2>& pipe, dim3& blocks, dim3& threads)
{
  bool loopc = false;
  do
  {
    Shared<int> retval = Shared<int>(1);
    HGAccumulator<int> _rv;
    *(retval.cpu_wr_ptr()) = 0;
    _rv.rv = retval.gpu_wr_ptr();
    pipe.out_wl().will_write();
    p_jump_roots <<<blocks, threads>>>(gg, pipe.in_wl(), pipe.out_wl(), _rv);
    cudaDeviceSynchronize();
    loopc = *(retval.cpu_rd_ptr()) > 0;
  }
  while (loopc);
}
__global__ void __launch_bounds__(__tb_gg_main_pipe_4_gpu_gb) gg_main_pipe_4_gpu_gb(CSRGraphTy gg, PipeContextT<Worklist2> pipe, int* retval, bool enable_lb, GlobalBarrier gb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  bool loopc = false;
  do
  {
    HGAccumulator<int> _rv;
    *retval = 0;
    _rv.rv = retval;
    gb.Sync();
    if (tid == 0)
      pipe.in_wl().reset_next_slot();
    p_jump_roots_dev (gg, pipe.in_wl(), pipe.out_wl(), _rv);
    _rv.local = *retval;
    gb.Sync();
    loopc = *retval > 0;
    gb.Sync();
  }
  while (loopc);
  gb.Sync();
  if (tid == 0)
  {
    pipe.save();
  }
}
__global__ void gg_main_pipe_4_gpu(CSRGraphTy gg, PipeContextT<Worklist2> pipe, dim3 blocks, dim3 threads, int* retval)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_one;
  bool loopc = false;
  do
  {
    HGAccumulator<int> _rv;
    *retval = 0;
    _rv.rv = retval;
    p_jump_roots <<<blocks, threads>>>(gg, pipe.in_wl(), pipe.out_wl(), _rv);
    cudaDeviceSynchronize();
    cudaDeviceSynchronize();
    loopc = *retval > 0;
  }
  while (loopc);
  if (tid == 0)
  {
    pipe.save();
  }
}
void gg_main_pipe_4_wrapper(CSRGraphTy& gg, PipeContextT<Worklist2>& pipe, dim3& blocks, dim3& threads)
{
  static GlobalBarrierLifetime gg_main_pipe_4_gpu_gb_barrier;
  static bool gg_main_pipe_4_gpu_gb_barrier_inited;
  extern bool enable_lb;
  static const size_t gg_main_pipe_4_gpu_gb_residency = maximum_residency(gg_main_pipe_4_gpu_gb, __tb_gg_main_pipe_4_gpu_gb, 0);
  static const size_t gg_main_pipe_4_gpu_gb_blocks = GG_MIN(blocks.x, ggc_get_nSM() * gg_main_pipe_4_gpu_gb_residency);
  if(!gg_main_pipe_4_gpu_gb_barrier_inited) { gg_main_pipe_4_gpu_gb_barrier.Setup(gg_main_pipe_4_gpu_gb_blocks); gg_main_pipe_4_gpu_gb_barrier_inited = true;};
  Shared<int> retval (1);
  if (enable_lb)
  {
    gg_main_pipe_4(gg,pipe,blocks,threads);
  }
  else
  {
    pipe.prep();

    // gg_main_pipe_4_gpu<<<1,1>>>(gg,pipe,blocks,threads,retval.gpu_wr_ptr(), enable_lb);
    gg_main_pipe_4_gpu_gb<<<gg_main_pipe_4_gpu_gb_blocks, __tb_gg_main_pipe_4_gpu_gb>>>(gg,pipe,retval.gpu_wr_ptr(), enable_lb, gg_main_pipe_4_gpu_gb_barrier);
    pipe.restore();
  }
}
void gg_main_pipe_3(CSRGraphTy& gg, dim3& blocks, dim3& threads)
{
  bool loopc = false;
  do
  {
    Shared<int> retval = Shared<int>(1);
    HGAccumulator<int> _rv;
    *(retval.cpu_wr_ptr()) = 0;
    _rv.rv = retval.gpu_wr_ptr();
    p_jump <<<blocks, threads>>>(gg, _rv);
    cudaDeviceSynchronize();
    loopc = *(retval.cpu_rd_ptr()) > 0;
  }
  while (loopc);
}
__global__ void __launch_bounds__(__tb_gg_main_pipe_3_gpu_gb) gg_main_pipe_3_gpu_gb(CSRGraphTy gg, int* retval, bool enable_lb, GlobalBarrier gb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  bool loopc = false;
  do
  {
    HGAccumulator<int> _rv;
    *retval = 0;
    _rv.rv = retval;
    gb.Sync();
    p_jump_dev (gg, _rv);
    _rv.local = *retval;
    gb.Sync();
    loopc = *retval > 0;
    gb.Sync();
  }
  while (loopc);
  gb.Sync();
}
__global__ void gg_main_pipe_3_gpu(CSRGraphTy gg, dim3 blocks, dim3 threads, int* retval)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_one;
  bool loopc = false;
  do
  {
    HGAccumulator<int> _rv;
    *retval = 0;
    _rv.rv = retval;
    p_jump <<<blocks, threads>>>(gg, _rv);
    cudaDeviceSynchronize();
    cudaDeviceSynchronize();
    loopc = *retval > 0;
  }
  while (loopc);
}
void gg_main_pipe_3_wrapper(CSRGraphTy& gg, dim3& blocks, dim3& threads)
{
  static GlobalBarrierLifetime gg_main_pipe_3_gpu_gb_barrier;
  static bool gg_main_pipe_3_gpu_gb_barrier_inited;
  extern bool enable_lb;
  static const size_t gg_main_pipe_3_gpu_gb_residency = maximum_residency(gg_main_pipe_3_gpu_gb, __tb_gg_main_pipe_3_gpu_gb, 0);
  static const size_t gg_main_pipe_3_gpu_gb_blocks = GG_MIN(blocks.x, ggc_get_nSM() * gg_main_pipe_3_gpu_gb_residency);
  if(!gg_main_pipe_3_gpu_gb_barrier_inited) { gg_main_pipe_3_gpu_gb_barrier.Setup(gg_main_pipe_3_gpu_gb_blocks); gg_main_pipe_3_gpu_gb_barrier_inited = true;};
  Shared<int> retval (1);
  if (enable_lb)
  {
    gg_main_pipe_3(gg,blocks,threads);
  }
  else
  {

    // gg_main_pipe_3_gpu<<<1,1>>>(gg,blocks,threads,retval.gpu_wr_ptr(), enable_lb);
    gg_main_pipe_3_gpu_gb<<<gg_main_pipe_3_gpu_gb_blocks, __tb_gg_main_pipe_3_gpu_gb>>>(gg,retval.gpu_wr_ptr(), enable_lb, gg_main_pipe_3_gpu_gb_barrier);
  }
}
void gg_main(CSRGraphTy& hg, CSRGraphTy& gg)
{
  dim3 blocks, threads;
  kernel_sizing(gg, blocks, threads);
  t_work.init_thread_work(gg.nnodes);
  int edge_blocks;
  int node_blocks;
  cudaEvent_t start;
  cudaEvent_t stop;
  PipeContextT<Worklist2> pipe;
  int it_hk = 1;
  Shared<index_type> edge_src (gg.nedges);
  Shared<char> edge_marks (gg.nedges);
  bool flag = false;
  edge_blocks = hg.nedges / TB_SIZE + 1;
  node_blocks = hg.nnodes / TB_SIZE + 1;
  edge_marks.zero_gpu();
  check_cuda(cudaEventCreate(&start));
  check_cuda(cudaEventCreate(&stop));
  check_cuda(cudaEventRecord(start));
  if (enable_lb)
  {
    t_work.reset_thread_work();
    Inspect_prep_edge_src <<<node_blocks, __tb_prep_edge_src>>>(gg, edge_src.gpu_wr_ptr(), t_work.thread_work_wl, t_work.thread_src_wl, enable_lb);
    cudaDeviceSynchronize();
    int num_items = t_work.thread_work_wl.in_wl().nitems();
    if (num_items != 0)
    {
      t_work.compute_prefix_sum();
      cudaDeviceSynchronize();
      prep_edge_src_TB_LB <<<node_blocks, __tb_prep_edge_src>>>(gg, edge_src.gpu_wr_ptr(), t_work.thread_prefix_work_wl.gpu_wr_ptr(), num_items, t_work.thread_src_wl);
      cudaDeviceSynchronize();
    }
  }
  prep_edge_src <<<node_blocks, __tb_prep_edge_src>>>(gg, edge_src.gpu_wr_ptr(), enable_lb);
  cudaDeviceSynchronize();
  check_cuda(cudaEventRecord(stop));
  init <<<node_blocks, threads>>>(gg);
  cudaDeviceSynchronize();
  hook_init <<<edge_blocks, threads>>>(gg, edge_src.gpu_rd_ptr());
  cudaDeviceSynchronize();
  gg_main_pipe_3_wrapper(gg,blocks,threads);
  pipe = PipeContextT<Worklist2>(gg.nnodes);
  {
    {
      do
      {
        pipe.out_wl().will_write();
        identify_roots <<<blocks, threads>>>(gg, pipe.in_wl(), pipe.out_wl());
        cudaDeviceSynchronize();
        pipe.in_wl().swap_slots();
        pipe.advance2();
        if (it_hk != 0)
        {
          Shared<int> retval = Shared<int>(1);
          HGAccumulator<int> _rv;
          *(retval.cpu_wr_ptr()) = 0;
          _rv.rv = retval.gpu_wr_ptr();
          hook_low_to_high <<<edge_blocks, threads>>>(gg, edge_src.gpu_rd_ptr(), edge_marks.gpu_wr_ptr(), _rv);
          cudaDeviceSynchronize();
          flag = *(retval.cpu_rd_ptr());
          it_hk = (it_hk + 1) % 4;
        }
        else
        {
          Shared<int> retval = Shared<int>(1);
          HGAccumulator<int> _rv;
          *(retval.cpu_wr_ptr()) = 0;
          _rv.rv = retval.gpu_wr_ptr();
          hook_high_to_low <<<edge_blocks, threads>>>(gg, edge_src.gpu_rd_ptr(), edge_marks.gpu_wr_ptr(), _rv);
          cudaDeviceSynchronize();
          flag = *(retval.cpu_rd_ptr());
        }
        if (!flag)
        {
          break;
        }
        gg_main_pipe_4_wrapper(gg,pipe,blocks,threads);
        p_jump_leaves <<<node_blocks, threads>>>(gg);
        cudaDeviceSynchronize();
      }
      while (flag);
    }
  }
  printf("iterations: %d\n", it_hk);
  Shared<int> count (1);
  *(count.cpu_wr_ptr()) = 0;
  count_components <<<blocks, threads>>>(gg, count.gpu_wr_ptr());
  cudaDeviceSynchronize();
  printf("components: %d\n", *(count.cpu_rd_ptr()));
  float ms =0;
  check_cuda(cudaEventElapsedTime(&ms, start, stop));
  DISCOUNT_TIME_NS = (int) (ms * 1000000);
  printf("prep_edge_src: %llu ns\n", DISCOUNT_TIME_NS);
}
