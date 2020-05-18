/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"
#include <curand.h>

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=False $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=1 $ instrument=set([]) $ unroll=[] $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=False $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=texture $ cuda.use_worklist_slots=True $ cuda.worklist_type=texture";
#include <curand.h>
#define UNMARKED 0
#define MARKED 1
#define NON_INDEPENDENT 2
#define NON_MAXIMAL 3
#define SEED1 0x12345678LL
#define SEED2 0xabbdef12LL
#define SEED3 0xcafe1234LL
#define SEED4 0x09832516LL
static const int __tb_one = 1;
__global__ void gen_prio_gpu(CSRGraph graph, unsigned int * prio, unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type node_end;
  x ^= tid;
  y ^= tid;
  z ^= tid;
  w ^= tid;
  assert(!(x == 0 && y == 0 && z == 0 && w == 0));
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    unsigned int t;
    t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    w = w ^ (w >> 19) ^ t ^ (t >> 8);
    prio[node] = w;
  }
}
void gen_prio(CSRGraph graph, unsigned int * prio)
{
  curandGenerator_t gen;
  check_rv(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937), CURAND_STATUS_SUCCESS);
  check_rv(curandSetPseudoRandomGeneratorSeed(gen, SEED1), CURAND_STATUS_SUCCESS);
  check_rv(curandSetGeneratorOrdering (gen, CURAND_ORDERING_PSEUDO_BEST), CURAND_STATUS_SUCCESS);
}
__global__ void init_wl(CSRGraph graph, WorklistT in_wl, WorklistT out_wl)
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
    (out_wl).push(node);
  }
}
__global__ void mark_nodes(CSRGraph graph, const unsigned int * __restrict__ prio, WorklistT in_wl, WorklistT out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type wlnode_end;
  wlnode_end = *((volatile index_type *) (in_wl).dindex);
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    bool pop;
    int node;
    index_type edge_end;
    pop = (in_wl).pop_id(wlnode, node);
    int max_prio = prio[node];
    int max_prio_node = node;
    edge_end = (graph).getFirstEdge((node) + 1);
    for (index_type edge = (graph).getFirstEdge(node) + 0; edge < edge_end; edge += 1)
    {
      index_type dst = graph.getAbsDestination(edge);
      if (dst != node && graph.node_data[dst] != NON_INDEPENDENT && prio[dst] >= max_prio)
      {
        if ((prio[dst] > max_prio) || dst > max_prio_node)
        {
          max_prio = prio[dst];
          max_prio_node = dst;
        }
      }
    }
    if (max_prio_node == node)
    {
      assert(graph.node_data[node] == UNMARKED);
      graph.node_data[node] = MARKED;
    }
  }
}
__global__ void drop_marked_nodes_and_nbors(CSRGraph graph, WorklistT in_wl, WorklistT out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type wlnode_end;
  wlnode_end = *((volatile index_type *) (in_wl).dindex);
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    bool pop;
    int node;
    pop = (in_wl).pop_id(wlnode, node);
    bool drop = false;
    if (graph.node_data[node] == MARKED)
    {
      drop = true;
    }
    if (!drop)
    {
      index_type edge_end;
      edge_end = (graph).getFirstEdge((node) + 1);
      for (index_type edge = (graph).getFirstEdge(node) + 0; edge < edge_end; edge += 1)
      {
        index_type dst = graph.getAbsDestination(edge);
        if (graph.node_data[dst] == MARKED)
        {
          drop = true;
        }
      }
    }
    if (!drop)
    {
      (out_wl).push(node);
    }
    else
    {
      if (graph.node_data[node] == UNMARKED)
      {
        graph.node_data[node] = NON_INDEPENDENT;
      }
    }
  }
}
void gg_main_pipe_1(CSRGraphTy& gg, int& STEPS, Shared<unsigned int>& prio, PipeContextT<WorklistT>& pipe, dim3& blocks, dim3& threads)
{
  {
    pipe.out_wl().will_write();
    init_wl <<<blocks, threads>>>(gg, pipe.in_wl(), pipe.out_wl());
    pipe.in_wl().swap_slots();
    pipe.advance2();
    while (pipe.in_wl().nitems())
    {
      pipe.out_wl().will_write();
      mark_nodes <<<blocks, threads>>>(gg, prio.gpu_rd_ptr(), pipe.in_wl(), pipe.out_wl());
      pipe.out_wl().will_write();
      drop_marked_nodes_and_nbors <<<blocks, threads>>>(gg, pipe.in_wl(), pipe.out_wl());
      pipe.in_wl().swap_slots();
      pipe.advance2();
      STEPS++;
    }
  }
}
__global__ void gg_main_pipe_1_gpu(CSRGraphTy gg, int STEPS, unsigned int* prio, PipeContextT<WorklistT> pipe, dim3 blocks, dim3 threads, int* cl_STEPS)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_one;
  STEPS = *cl_STEPS;
  {
    init_wl <<<blocks, threads>>>(gg, pipe.in_wl(), pipe.out_wl());
    pipe.in_wl().swap_slots();
    cudaDeviceSynchronize();
    pipe.advance2();
    while (pipe.in_wl().nitems())
    {
      mark_nodes <<<blocks, threads>>>(gg, prio, pipe.in_wl(), pipe.out_wl());
      cudaDeviceSynchronize();
      drop_marked_nodes_and_nbors <<<blocks, threads>>>(gg, pipe.in_wl(), pipe.out_wl());
      pipe.in_wl().swap_slots();
      cudaDeviceSynchronize();
      pipe.advance2();
      STEPS++;
    }
  }
  if (tid == 0)
  {
    *cl_STEPS = STEPS;
  }
}
void gg_main_pipe_1_wrapper(CSRGraphTy& gg, int& STEPS, Shared<unsigned int>& prio, PipeContextT<WorklistT>& pipe, dim3& blocks, dim3& threads)
{
  if (false)
  {
    gg_main_pipe_1(gg,STEPS,prio,pipe,blocks,threads);
  }
  else
  {
    int* cl_STEPS;
    check_cuda(cudaMalloc(&cl_STEPS, sizeof(int) * 1));
    check_cuda(cudaMemcpy(cl_STEPS, &STEPS, sizeof(int) * 1, cudaMemcpyHostToDevice));

    gg_main_pipe_1_gpu<<<1,1>>>(gg,STEPS,prio.gpu_wr_ptr(),pipe,blocks,threads,cl_STEPS);
    // gg_main_pipe_1_gpu_gb<<<gg_main_pipe_1_gpu_gb_blocks, __tb_gg_main_pipe_1_gpu_gb>>>(gg,STEPS,prio.gpu_wr_ptr(),pipe,cl_STEPS, gg_main_pipe_1_gpu_gb_barrier);
    check_cuda(cudaMemcpy(&STEPS, cl_STEPS, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    check_cuda(cudaFree(cl_STEPS));
  }
}
void gg_main(CSRGraphTy& hg, CSRGraphTy& gg)
{
  dim3 blocks, threads;
  kernel_sizing(gg, blocks, threads);
  PipeContextT<WorklistT> pipe;
  Shared<unsigned int> prio (hg.nnodes);
  int STEPS = 0;
  ggc::Timer t ("random");
  t.start();
  gen_prio_gpu <<<blocks, threads>>>(gg, prio.gpu_wr_ptr(), SEED1, SEED2, SEED3, SEED4);
  cudaDeviceSynchronize();
  t.stop();
  printf("Random number generation took %llu ns\n", t.duration());
  pipe = PipeContextT<WorklistT>(gg.nnodes);
  gg_main_pipe_1_wrapper(gg,STEPS,prio,pipe,blocks,threads);
  printf("Total steps: %d\n", STEPS);
}
