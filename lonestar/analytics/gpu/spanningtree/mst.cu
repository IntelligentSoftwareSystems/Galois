/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "thread_work.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=False $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=1 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=False $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
AppendOnlyList el;
#include "mst.h"
#define INF UINT_MAX
const int DEBUG = 0;
static const int __tb_union_components = TB_SIZE;
__global__ void init_wl(CSRGraph graph, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type node_end;
  // FP: "1 -> 2;
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    (out_wl).push(node);
  }
  // FP: "4 -> 5;
}
__global__ void find_comp_min_elem(CSRGraph graph, struct comp_data comp, LockArrayTicket complocks, ComponentSpace cs, int level, AppendOnlyList bosses, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type wlnode_end;
  // FP: "1 -> 2;

  // FP: "2 -> 3;
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
    bool isBoss = srccomp == node;
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
    if (isBoss && degree)
    {
      bosses.push(node);
    }
    if (minwt != INF)
    {
      (out_wl).push(node);
      {
        #if __CUDACC_VER_MAJOR__ >= 7
        volatile bool done_ = false;
        #else
        bool done_ = false;
        #endif
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
      if (isBoss && degree)
      {
        (out_wl).push(node);
      }
    }
  }
  // FP: "30 -> 31;
}
__global__ void union_components(CSRGraph graph, ComponentSpace cs, struct comp_data compdata, int level, AppendOnlyList el, AppendOnlyList ew, AppendOnlyList b_in, AppendOnlyList b_out, Worklist2 in_wl, Worklist2 out_wl, GlobalBarrier gb, HGAccumulator<int> ret_val)
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
  index_type wlnode_rup;
  // FP: "1 -> 2;
  wlnode_end = *((volatile index_type *) (b_in).dindex);
  wlnode_rup = ((0) + roundup(((*((volatile index_type *) (b_in).dindex)) - (0)), (nthreads)));
  for (index_type wlnode = 0 + tid; wlnode < wlnode_rup; wlnode += nthreads)
  {
    int node;
    bool pop;
    pop = (b_in).pop_id(wlnode, node);
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
        b_out.push(node);
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
void gg_main(CSRGraph& hg, CSRGraph& gg)
{
  dim3 blocks, threads;
  kernel_sizing(gg, blocks, threads);
  static GlobalBarrierLifetime union_components_barrier;
  static bool union_components_barrier_inited;
  struct comp_data comp;
  PipeContextT<Worklist2> pipe;
  // FP: "1 -> 2;
  ComponentSpace cs (hg.nnodes);
  // FP: "2 -> 3;
  el = AppendOnlyList(hg.nedges);
  // FP: "3 -> 4;
  AppendOnlyList ew (hg.nedges);
  // FP: "4 -> 5;
  AppendOnlyList bosses[2] = {AppendOnlyList(hg.nnodes), AppendOnlyList(hg.nnodes)};
  int cur_boss = 0;
  // FP: "5 -> 6;
  static const size_t union_components_residency = maximum_residency(union_components, __tb_union_components, 0);
  static const size_t union_components_blocks = GG_MIN(blocks.x, ggc_get_nSM() * union_components_residency);
  if(!union_components_barrier_inited) { union_components_barrier.Setup(union_components_blocks); union_components_barrier_inited = true;};
  // FP: "6 -> 7;
  // FP: "7 -> 8;
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
  // FP: "8 -> 9;
  LockArrayTicket complocks (hg.nnodes);
  // FP: "9 -> 10;
  int level = 1;
  int mw = 0;
  int last_mw = 0;
  // FP: "10 -> 11;
  pipe = PipeContextT<Worklist2>(hg.nnodes);
  {
    {
      pipe.out_wl().will_write();
      init_wl <<<blocks, threads>>>(gg, pipe.in_wl(), pipe.out_wl());
      pipe.in_wl().swap_slots();
      pipe.advance2();
      // FP: "12 -> 13;
      while (pipe.in_wl().nitems())
      {
        bool loopc = false;
        last_mw = mw;
        pipe.out_wl().will_write();
        find_comp_min_elem <<<blocks, threads>>>(gg, comp, complocks, cs, level, bosses[cur_boss], pipe.in_wl(), pipe.out_wl());
        pipe.in_wl().swap_slots();
        pipe.advance2();
        do
        {
          Shared<int> retval = Shared<int>(1);
          HGAccumulator<int> _rv;
          *(retval.cpu_wr_ptr()) = 0;
          _rv.rv = retval.gpu_wr_ptr();
          pipe.out_wl().will_write();
          union_components <<<union_components_blocks, __tb_union_components>>>(gg, cs, comp, level, el, ew, bosses[cur_boss], bosses[cur_boss ^ 1], pipe.in_wl(), pipe.out_wl(), union_components_barrier, _rv);
          loopc = *(retval.cpu_rd_ptr()) > 0;
          cur_boss ^= 1;
          bosses[cur_boss].reset();
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
  pipe.free();
  unsigned long int rweight = 0;
  size_t nmstedges ;
  nmstedges = ew.nitems();
  printf("nmstedges = %d\n", nmstedges);
  int *h_list = (int *)malloc(nmstedges*sizeof(int));
  check_cuda(cudaMemcpy(h_list, ew.list.gpu_rd_ptr(), nmstedges * sizeof(int), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nmstedges; i ++) rweight += h_list[i];
  printf("final mstwt: %llu\n", rweight);
  printf("total edges: %llu, total components: %llu\n", nmstedges, cs.numberOfComponentsHost());
}
