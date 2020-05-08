/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "thread_work.h"

mgpu::standard_context_t context;

void kernel_sizing(CSRGraphTex &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=False $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=1 $ instrument=set([]) $ unroll=[] $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=False $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=texture $ cuda.use_worklist_slots=True $ cuda.worklist_type=texture";
AppendOnlyList el;
#include "mst.h"
#define INF UINT_MAX
const int DEBUG = 0;
static const int __tb_union_components = TB_SIZE;
__global__ void init_wl(CSRGraphTex graph, WorklistT in_wl, WorklistT out_wl)
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
__global__ void find_comp_min_elem(CSRGraphTex graph, struct comp_data comp, LockArrayTicket complocks, ComponentSpace cs, int level, WorklistT in_wl, WorklistT out_wl)
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
    int node;
    bool pop;
    index_type edge_end;
    pop = (in_wl).pop_id(wlnode, node);
    unsigned minwt = INF;
    unsigned minedge = INF;
    int degree = graph.getOutDegree(node);
    int mindstcomp  = 0;
    int srccomp = cs.find(node);
    edge_end = (graph).getFirstEdge((node) + 1);
    for (index_type edge = (graph).getFirstEdge(node) + 0; edge < edge_end; edge += 1)
    {
      int edgewt = graph.getAbsWeight(edge);
      if (edgewt < minwt)
      {
        int dstcomp = cs.find(graph.getAbsDestination(edge));
        if (dstcomp != srccomp)
        {
          minwt = edgewt;
          minedge = edge;
        }
      }
    }
    if (minwt != INF)
    {
      (out_wl).push(node);
      {
        volatile bool done_ = false;
		int _ticket = (complocks).reserve(srccomp);
        while (!done_)
        {
          if (complocks.acquire_or_fail(srccomp, _ticket))
          {
            if (comp.minwt[srccomp] == 0 || (comp.lvl[srccomp] < level) || (comp.minwt[srccomp] > minwt))
            {
              comp.minwt[srccomp] = minwt;
              comp.lvl[srccomp] = level;
              comp.minedge[srccomp] = minedge;
            }
            complocks.release(srccomp);
            done_ = true;
          }
        }
      }
    }
    else
    {
      if (cs.isBoss(node) && degree)
      {
        (out_wl).push(node);
      }
    }
  }
}
__global__ void union_components(CSRGraphTex graph, ComponentSpace cs, struct comp_data compdata, int level, AppendOnlyList el, AppendOnlyList ew, WorklistT in_wl, WorklistT out_wl, GlobalBarrier gb, HGAccumulator<int> ret_val)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  typedef cub::BlockReduce<int, TB_SIZE> _br;
  __shared__ _br::TempStorage _ts;
  ret_val.thread_entry();
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type wlnode_end;
  wlnode_end = roundup((*((volatile index_type *) (in_wl).dindex)), (nthreads));
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    int node;
    bool pop;
    pop = (in_wl).pop_id(wlnode, node);
    int r = 0;
    int dstcomp = -1;
    int srccomp = -1;
    if (pop && compdata.lvl[node] == level)
    {
      srccomp = cs.find(node);
      dstcomp = cs.find(graph.getAbsDestination(compdata.minedge[node]));
    }
    gb.Sync();
    if (srccomp != dstcomp)
    {
      if (!cs.unify(srccomp, dstcomp))
      {
        r = 1;
      }
      else
      {
        el.push(compdata.minedge[node]);
        ew.push(compdata.minwt[node]);
      }
    }
    gb.Sync();
    if (r)
    {
      ret_val.reduce(true);
      continue;
    }
  }
  ret_val.thread_exit<_br>(_ts);
}
void gg_main(CSRGraphTex& hg, CSRGraphTex& gg)
{
  dim3 blocks, threads;
  kernel_sizing(gg, blocks, threads);
  static GlobalBarrierLifetime union_components_barrier;
  static bool union_components_barrier_inited;
  struct comp_data comp;
  PipeContextT<WorklistT> pipe;
  ComponentSpace cs (hg.nnodes);
  el = AppendOnlyList(hg.nedges);
  AppendOnlyList ew (hg.nedges);
  static const size_t union_components_residency = maximum_residency(union_components, __tb_union_components, 0);
  static const size_t union_components_blocks = GG_MIN(blocks.x, ggc_get_nSM() * union_components_residency);
  if(!union_components_barrier_inited) { union_components_barrier.Setup(union_components_blocks); union_components_barrier_inited = true;};
  comp.weight.alloc(hg.nnodes);
  comp.edge.alloc(hg.nnodes);
  comp.node.alloc(hg.nnodes);
  comp.level.alloc(hg.nnodes);
  comp.dstcomp.alloc(hg.nnodes);
  comp.lvl = comp.level.zero_gpu();
  comp.minwt = comp.weight.zero_gpu();
  comp.minedge = comp.edge.gpu_wr_ptr();
  comp.minnode = comp.node.gpu_wr_ptr();
  comp.mindstcomp = comp.dstcomp.gpu_wr_ptr();
  LockArrayTicket complocks (hg.nnodes);
  int level = 1;
  int mw = 0;
  int last_mw = 0;
  pipe = PipeContextT<WorklistT>(hg.nnodes);
  {
    {
      pipe.out_wl().will_write();
      init_wl <<<blocks, threads>>>(gg, pipe.in_wl(), pipe.out_wl());
      pipe.in_wl().swap_slots();
      pipe.advance2();
      while (pipe.in_wl().nitems())
      {
        bool loopc = false;
        last_mw = mw;
        pipe.out_wl().will_write();
        find_comp_min_elem <<<blocks, threads>>>(gg, comp, complocks, cs, level, pipe.in_wl(), pipe.out_wl());
        pipe.in_wl().swap_slots();
        pipe.advance2();
        do
        {
          Shared<int> retval = Shared<int>(1);
          HGAccumulator<int> _rv;
          *(retval.cpu_wr_ptr()) = 0;
          _rv.rv = retval.gpu_wr_ptr();
          pipe.out_wl().will_write();
          union_components <<<union_components_blocks, __tb_union_components>>>(gg, cs, comp, level, el, ew, pipe.in_wl(), pipe.out_wl(), union_components_barrier, _rv);
          loopc = *(retval.cpu_rd_ptr()) > 0;
        }
        while (loopc);
        mw = el.nitems();
        level++;
        if (last_mw == mw)
        {
          break;
        }
      }
    }
  }
  unsigned long int rweight = 0;
  size_t nmstedges ;
  nmstedges = ew.nitems();
  mgpu::reduce(ew.list.gpu_rd_ptr(), nmstedges, &rweight, mgpu::plus_t<long unsigned int>(), context);
  printf("number of iterations: %d\n", level);
  printf("final mstwt: %llu\n", rweight);
  printf("total edges: %llu, total components: %llu\n", nmstedges, cs.numberOfComponentsHost());
}
