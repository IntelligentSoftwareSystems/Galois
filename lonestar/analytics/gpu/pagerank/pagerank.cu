/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "thread_work.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=True $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['wp', 'fg']) $ cc_disable=set([]) $ tb_lb=False $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
struct ThreadWork t_work;
bool enable_lb = false;
typedef int edge_data_type;
typedef int node_data_type;
typedef float* gfloat_p;
float* P_CURR ;
float* P_NEXT ;
extern const float ALPHA = 0.85;
extern const float EPSILON = 0.000001;
extern int MAX_ITERATIONS ;
static const int __tb_gg_main_pipe_1_gpu_gb = 256;
static const int __tb_pagerank_main = TB_SIZE;
static const int __tb_remove_dups = TB_SIZE;
__global__ void init_1(CSRGraph graph, float * p_curr, float * residual)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  index_type node_end;
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    float update;
    index_type edge_end;
    p_curr[node] = 1.0 - ALPHA;
    update = 1.0/graph.getOutDegree(node);
    edge_end = (graph).getFirstEdge((node) + 1);
    for (index_type edge = (graph).getFirstEdge(node) + 0; edge < edge_end; edge += 1)
    {
      index_type dst;
      dst = graph.getAbsDestination(edge);
      atomicAdd(residual + dst, update);
    }
  }
}
__device__ void init_2_dev(CSRGraph graph, float * residual, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  index_type _start_22;
  index_type node_end;
  _start_22 = (out_wl).push_range((tid < ((graph).nnodes)) ? ((((graph).nnodes) - 1 - tid)/nthreads + 1) : 0);;
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid, node_pos = 0; node < node_end; node_pos++, node += nthreads)
  {
    residual[node] *= (1.0 - ALPHA) * ALPHA;
    (out_wl).do_push(_start_22, node_pos, node);
  }
}
__global__ void init_2(CSRGraph graph, float * residual, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;

  if (tid == 0)
    in_wl.reset_next_slot();

  init_2_dev(graph, residual, in_wl, out_wl);
}
__global__ void remove_dups(int * marks, Worklist2 in_wl, Worklist2 out_wl, GlobalBarrier gb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  if (tid == 0)
    in_wl.reset_next_slot();

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
      index_type _start_37;
      _start_37 = (out_wl).setup_push_warp_one();;
      (out_wl).do_push(_start_37, 0, node);
    }
  }
}
__device__ void pagerank_main_dev(CSRGraph graph, float * p_curr, float * residual, float * p_diff, bool enable_lb, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_pagerank_main;
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
    int sdeg;
    float update;
    int node;
    bool pop;
    float res;
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    pop = (in_wl).pop_id(wlnode, node);
    if (pop)
    {
      res =atomicExch(residual + node, 0);
      p_curr[node] += res;
      sdeg = graph.getOutDegree(node);
      update = res * ALPHA / sdeg;
    }
    struct NPInspector1 _np = {0,0,0,0,0,0};
    __shared__ struct { float update; } _np_closure [TB_SIZE];
    _np_closure[threadIdx.x].update = update;
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
        update = _np_closure[nps.warp.src[warpid]].update;
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type edge;
          edge = _np_w_start +_np_ii;
          {
            index_type dst;
            float prev;
            dst = graph.getAbsDestination(edge);
            prev = atomicAdd(residual + dst, update);
            if (prev + update > EPSILON && prev < EPSILON)
            {
              index_type _start_57;
              _start_57 = (out_wl).setup_push_warp_one();;
              (out_wl).do_push(_start_57, 0, dst);
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
        update = _np_closure[nps.fg.src[_np_i]].update;
        edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          float prev;
          dst = graph.getAbsDestination(edge);
          prev = atomicAdd(residual + dst, update);
          if (prev + update > EPSILON && prev < EPSILON)
          {
            index_type _start_57;
            _start_57 = (out_wl).setup_push_warp_one();;
            (out_wl).do_push(_start_57, 0, dst);
          }
        }
      }
      _np.execute_round_done(ITSIZE);
      __syncthreads();
    }
    assert(threadIdx.x < __kernel_tb_size);
    update = _np_closure[threadIdx.x].update;
  }
}
__global__ void __launch_bounds__(TB_SIZE, 3) pagerank_main(CSRGraph graph, float * p_curr, float * residual, float * p_diff, bool enable_lb, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;

  if (tid == 0)
    in_wl.reset_next_slot();

  pagerank_main_dev(graph, p_curr, residual, p_diff, enable_lb, in_wl, out_wl);
}
void gg_main_pipe_1(gfloat_p p2, gfloat_p p0, gfloat_p rp, int& iter, CSRGraph& gg, CSRGraph& hg, int MAX_ITERATIONS, PipeContextT<Worklist2>& pipe, dim3& blocks, dim3& threads)
{
  {
    pipe.out_wl().will_write();
    init_2 <<<blocks, threads>>>(gg, rp, pipe.in_wl(), pipe.out_wl());
    cudaDeviceSynchronize();
    pipe.in_wl().swap_slots();
    pipe.advance2();
    while (pipe.in_wl().nitems())
    {
      pipe.out_wl().will_write();
      pagerank_main <<<blocks, __tb_pagerank_main>>>(gg, p0, rp, p2, enable_lb, pipe.in_wl(), pipe.out_wl());
      cudaDeviceSynchronize();
      pipe.in_wl().swap_slots();
      pipe.advance2();
      iter++;
      if (iter >= MAX_ITERATIONS)
      {
        break;
      }
    }
  }
}
__global__ void __launch_bounds__(__tb_gg_main_pipe_1_gpu_gb) gg_main_pipe_1_gpu_gb(gfloat_p p2, gfloat_p p0, gfloat_p rp, int iter, CSRGraph gg, CSRGraph hg, int MAX_ITERATIONS, PipeContextT<Worklist2> pipe, int* cl_iter, bool enable_lb, GlobalBarrier gb)
{
  unsigned tid = TID_1D;

  iter = *cl_iter;
  {
    if (tid == 0)
      pipe.in_wl().reset_next_slot();
    init_2_dev (gg, rp, pipe.in_wl(), pipe.out_wl());
    pipe.in_wl().swap_slots();
    gb.Sync();
    pipe.advance2();
    while (pipe.in_wl().nitems())
    {
      if (tid == 0)
        pipe.in_wl().reset_next_slot();
      pagerank_main_dev (gg, p0, rp, p2, enable_lb, pipe.in_wl(), pipe.out_wl());
      pipe.in_wl().swap_slots();
      gb.Sync();
      pipe.advance2();
      iter++;
      if (iter >= MAX_ITERATIONS)
      {
        break;
      }
    }
  }
  gb.Sync();
  if (tid == 0)
  {
    *cl_iter = iter;
  }
}
__global__ void gg_main_pipe_1_gpu(gfloat_p p2, gfloat_p p0, gfloat_p rp, int iter, CSRGraph gg, CSRGraph hg, int MAX_ITERATIONS, PipeContextT<Worklist2> pipe, dim3 blocks, dim3 threads, int* cl_iter, bool enable_lb)
{
  unsigned tid = TID_1D;

  iter = *cl_iter;
  {
    init_2 <<<blocks, threads>>>(gg, rp, pipe.in_wl(), pipe.out_wl());
    cudaDeviceSynchronize();
    pipe.in_wl().swap_slots();
    cudaDeviceSynchronize();
    pipe.advance2();
    while (pipe.in_wl().nitems())
    {
      pagerank_main <<<blocks, __tb_pagerank_main>>>(gg, p0, rp, p2, enable_lb, pipe.in_wl(), pipe.out_wl());
      cudaDeviceSynchronize();
      pipe.in_wl().swap_slots();
      cudaDeviceSynchronize();
      pipe.advance2();
      iter++;
      if (iter >= MAX_ITERATIONS)
      {
        break;
      }
    }
  }
  if (tid == 0)
  {
    *cl_iter = iter;
  }
}
void gg_main_pipe_1_wrapper(gfloat_p p2, gfloat_p p0, gfloat_p rp, int& iter, CSRGraph& gg, CSRGraph& hg, int MAX_ITERATIONS, PipeContextT<Worklist2>& pipe, dim3& blocks, dim3& threads)
{
  static GlobalBarrierLifetime gg_main_pipe_1_gpu_gb_barrier;
  static bool gg_main_pipe_1_gpu_gb_barrier_inited;
  extern bool enable_lb;
  static const size_t gg_main_pipe_1_gpu_gb_residency = maximum_residency(gg_main_pipe_1_gpu_gb, __tb_gg_main_pipe_1_gpu_gb, 0);
  static const size_t gg_main_pipe_1_gpu_gb_blocks = GG_MIN(blocks.x, ggc_get_nSM() * gg_main_pipe_1_gpu_gb_residency);
  if(!gg_main_pipe_1_gpu_gb_barrier_inited) { gg_main_pipe_1_gpu_gb_barrier.Setup(gg_main_pipe_1_gpu_gb_blocks); gg_main_pipe_1_gpu_gb_barrier_inited = true;};
  if (enable_lb)
  {
    gg_main_pipe_1(p2,p0,rp,iter,gg,hg,MAX_ITERATIONS,pipe,blocks,threads);
  }
  else
  {
    int* cl_iter;
    check_cuda(cudaMalloc(&cl_iter, sizeof(int) * 1));
    check_cuda(cudaMemcpy(cl_iter, &iter, sizeof(int) * 1, cudaMemcpyHostToDevice));

    // gg_main_pipe_1_gpu<<<1,1>>>(p2,p0,rp,iter,gg,hg,MAX_ITERATIONS,pipe,blocks,threads,cl_iter, enable_lb);
    gg_main_pipe_1_gpu_gb<<<gg_main_pipe_1_gpu_gb_blocks, __tb_gg_main_pipe_1_gpu_gb>>>(p2,p0,rp,iter,gg,hg,MAX_ITERATIONS,pipe,cl_iter, enable_lb, gg_main_pipe_1_gpu_gb_barrier);
    check_cuda(cudaMemcpy(&iter, cl_iter, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    check_cuda(cudaFree(cl_iter));
  }
}
void gg_main(CSRGraph& hg, CSRGraph& gg)
{
  dim3 blocks, threads;
  kernel_sizing(gg, blocks, threads);
  t_work.init_thread_work(gg.nnodes);
  static GlobalBarrierLifetime remove_dups_barrier;
  static bool remove_dups_barrier_inited;
  PipeContextT<Worklist2> pipe;
  Shared<float> p[3] = {Shared<float> (hg.nnodes), Shared<float> (hg.nnodes), Shared<float>(hg.nnodes)};
  Shared<float> r (hg.nnodes);
  Shared<int> marks (hg.nnodes);
  static const size_t remove_dups_residency = maximum_residency(remove_dups, __tb_remove_dups, 0);
  static const size_t remove_dups_blocks = GG_MIN(blocks.x, ggc_get_nSM() * remove_dups_residency);
  if(!remove_dups_barrier_inited) { remove_dups_barrier.Setup(remove_dups_blocks); remove_dups_barrier_inited = true;};
  int iter = 0;
  r.zero_gpu();
  init_1 <<<blocks, threads>>>(gg, p[0].gpu_wr_ptr(), r.gpu_wr_ptr());
  cudaDeviceSynchronize();
  gfloat_p p0 =p[0].gpu_wr_ptr();
  gfloat_p p2 =p[2].gpu_wr_ptr();
  gfloat_p rp =r.gpu_wr_ptr();
  pipe = PipeContextT<Worklist2>(hg.nedges);
  gg_main_pipe_1_wrapper(p2,p0,rp,iter,gg,hg,MAX_ITERATIONS,pipe,blocks,threads);
  printf("PR took %d iterations\n", iter);
  P_CURR = p[0].cpu_rd_ptr();
  P_NEXT = p[0].cpu_rd_ptr();
}
