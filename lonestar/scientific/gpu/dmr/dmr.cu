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

void kernel_sizing(CSRGraphTex &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=False $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=1 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=False $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=texture $ cuda.use_worklist_slots=True $ cuda.worklist_type=texture";
#define CAVLEN 256
#define BCLEN 1024
#include "dmrggc.inc"
static const int __tb_refine = TB_SIZE;
__global__ void check_triangles(Mesh mesh, unsigned int * bad_triangles, int start, WorklistT in_wl, WorklistT out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  //const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  index_type ele_end;
  // FP: "1 -> 2;
  uint3* el ;
  int count = 0;
  // FP: "2 -> 3;
  ele_end = ((mesh).nelements);
  for (index_type ele = start + tid; ele < ele_end; ele += nthreads)
  {
    if (ele < mesh.nelements)
    {
      if (!(mesh.isdel[ele] || IS_SEGMENT(mesh.elements[ele])))
      {
        if (!mesh.isbad[ele])
        {
          el = &mesh.elements[ele];
          mesh.isbad[ele] = (angleLT(mesh, el->x, el->y, el->z) || angleLT(mesh, el->z, el->x, el->y) || angleLT(mesh, el->y, el->z, el->x));
        }
        if (mesh.isbad[ele])
        {
          count++;
          (out_wl).push(ele);
        }
      }
    }
  }
  // FP: "15 -> 16;
  atomicAdd(bad_triangles, count);
  // FP: "16 -> 17;
}
__global__ void __launch_bounds__(TB_SIZE) refine(Mesh mesh, int debg, uint * nnodes, uint * nelements, WorklistT in_wl, WorklistT out_wl, WorklistT re_wl, ExclusiveLocks _ex, GlobalBarrier gb)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  //const unsigned __kernel_tb_size = TB_SIZE;
  if (tid == 0)
    in_wl.reset_next_slot();

  uint cavity[CAVLEN] ;
  uint nc = 0;
  uint boundary[BCLEN] ;
  uint bc = 0;
  bool repush = false;
  index_type wlele_end = *((volatile index_type *) (in_wl).dindex);
  index_type wlele_rup = ((0) + roundup(((*((volatile index_type *) (in_wl).dindex)) - (0)), (nthreads)));
  index_type wlele_block_size = wlele_rup / nthreads;
  index_type wlele_block_start = (0 + tid) * wlele_block_size;
  int stage = 0;
  for (index_type wlele = wlele_block_start; wlele < (wlele_block_start + wlele_block_size) && (wlele < wlele_rup); wlele++)
  {
    FORD cx;
    FORD cy;
    bool pop;
    int ele;
    nc = 0;
    bc = 0;
    repush = false;
    stage = 0;
    pop = (in_wl).pop_id(wlele, ele);
    if (pop && ele < mesh.nelements && mesh.isbad[ele] && !mesh.isdel[ele])
    {
      uint oldcav;
      cavity[nc++] = ele;
      do
      {
        oldcav = cavity[0];
        cavity[0] = opposite(mesh, ele);
      }
      while (cavity[0] != oldcav);
      if (!build_cavity(mesh, cavity, nc, CAVLEN, boundary, bc, cx, cy))
      {
        build_cavity(mesh, cavity, nc, CAVLEN, boundary, bc, cx, cy);
      }
    }
    int nodes_added = 0;
    int elems_added = 0;
    {
      _ex.mark_p1(nc, (int *) cavity, tid);
      _ex.mark_p1_iterator(2, bc, 4, (int *) boundary, tid);
      gb.Sync();
      _ex.mark_p2(nc, (int *) cavity, tid);
      _ex.mark_p2_iterator(2, bc, 4, (int *) boundary, tid);
      gb.Sync();
      int _x = 1;
      _x &= _ex.owns(nc, (int *) cavity, tid);
      _x &= _ex.owns_iterator(2, bc, 4, (int *) boundary, tid);
      if (_x)
      {
        if (nc > 0)
        {
          nodes_added = 1;
          elems_added = (bc >> 2) + (IS_SEGMENT(mesh.elements[cavity[0]]) ? 2 : 0);
          uint cnode ;
          uint cseg1 = 0;
          uint cseg2 = 0;
          uint nelements_added ;
          uint oldelements ;
          uint newelemndx ;
          cnode = add_node(mesh, cx, cy, atomicAdd(nnodes, 1));
          nelements_added = elems_added;
          oldelements = atomicAdd(nelements, nelements_added);
          newelemndx = oldelements;
          if (IS_SEGMENT(mesh.elements[cavity[0]]))
          {
            cseg1 = add_segment(mesh, mesh.elements[cavity[0]].x, cnode, newelemndx++);
            cseg2 = add_segment(mesh, cnode, mesh.elements[cavity[0]].y, newelemndx++);
          }
          for (int i = 0; i < bc; i+=4)
          {
            uint ntri = add_triangle(mesh, boundary[i], boundary[i+1], cnode, boundary[i+2], boundary[i+3], newelemndx++);
          }
          assert(oldelements + nelements_added == newelemndx);
          setup_neighbours(mesh, oldelements, newelemndx);
          repush = true;
          for (int i = 0; i < nc; i++)
          {
            mesh.isdel[cavity[i]] = true;
            if (cavity[i] == ele)
            {
              repush = false;
            }
          }
        }
      }
      else
      {
        repush = true;
      }
    }
    gb.Sync();
    if (repush)
    {
      (out_wl).push(ele);
      continue;
    }
  }
}
void refine_mesh(ShMesh& mesh, dim3 blocks, dim3 threads)
{
  ExclusiveLocks refine_ex_locks(mesh.maxnelements);
  static GlobalBarrierLifetime refine_barrier;
  static bool refine_barrier_inited;
  PipeContextT<WorklistT> pipe;
  // FP: "1 -> 2;
  Shared<uint> nbad (1);
  Mesh gmesh (mesh);
  Shared<uint> nelements (1);
  Shared<uint> nnodes (1);
  //int cnbad ;
  bool orig = false;
  ggc::Timer t ("total");
  // FP: "2 -> 3;
  // FP: "3 -> 4;
  static const size_t refine_residency = maximum_residency(refine, __tb_refine, 0);
  static const size_t refine_blocks = GG_MIN(blocks.x, ggc_get_nSM() * refine_residency);
  if(!refine_barrier_inited) { refine_barrier.Setup(refine_blocks); refine_barrier_inited = true;};
  // FP: "4 -> 5;
  find_neighbours_cpu(mesh);
  gmesh.refresh(mesh);
  *(nelements.cpu_wr_ptr(true)) = mesh.nelements;
  *(nnodes.cpu_wr_ptr(true)) = mesh.nnodes;
  // FP: "5 -> 6;
  pipe = PipeContextT<WorklistT>(mesh.nelements);
  {
    {
      int lastnelements = 0;
      // FP: "7 -> 8;
      *(nbad.cpu_wr_ptr(true)) = 0;
      t.start();
      // FP: "8 -> 9;
      pipe.out_wl().will_write();
      check_triangles <<<blocks, threads>>>(gmesh, nbad.gpu_wr_ptr(), 0, pipe.in_wl(), pipe.out_wl());
      pipe.in_wl().swap_slots();
      pipe.advance2();
      // FP: "9 -> 10;
      printf("%d initial bad triangles\n", *(nbad.cpu_rd_ptr()) );;
      // FP: "10 -> 11;
      while (pipe.in_wl().nitems())
      {
        lastnelements = gmesh.nelements;
        {
          pipe.out_wl().will_write();
          pipe.re_wl().will_write();
          refine <<<refine_blocks, __tb_refine>>>(gmesh, 32, nnodes.gpu_wr_ptr(), nelements.gpu_wr_ptr(), pipe.in_wl(), pipe.out_wl(), pipe.re_wl(), refine_ex_locks, refine_barrier);
          pipe.in_wl().swap_slots();
          pipe.retry2();
        }
        gmesh.nnodes = mesh.nnodes = *(nnodes.cpu_rd_ptr());
        gmesh.nelements = mesh.nelements = *(nelements.cpu_rd_ptr());
        *(nbad.cpu_wr_ptr(true)) = 0;
        printf("checking triangles ...\n");
        pipe.out_wl().will_write();
        if (orig)
          check_triangles_orig <<<blocks, threads>>>(gmesh, nbad.gpu_wr_ptr(), lastnelements, pipe.in_wl(), pipe.out_wl());
        else
          check_triangles <<<blocks, threads>>>(gmesh, nbad.gpu_wr_ptr(), lastnelements, pipe.in_wl(), pipe.out_wl());
        pipe.in_wl().swap_slots();
        pipe.advance2();
        printf("%d bad triangles\n", *(nbad.cpu_rd_ptr()) );
      }
      // FP: "18 -> 19;
      t.stop();
      printf("time: %llu ns\n", t.duration());
      // FP: "19 -> 20;
      {
        *(nbad.cpu_wr_ptr(true)) = 0;
        // FP: "21 -> 22;
        pipe.out_wl().will_write();
        check_triangles <<<blocks, threads>>>(gmesh, nbad.gpu_wr_ptr(), 0, pipe.in_wl(), pipe.out_wl());
        pipe.in_wl().swap_slots();
        pipe.advance2();
        // FP: "22 -> 23;
        printf("%d (%d) final bad triangles\n", *(nbad.cpu_rd_ptr()), pipe.in_wl().nitems() );
        // FP: "23 -> 24;
      }
      // FP: "20 -> 21;
    }
  }
  pipe.free();
  // FP: "6 -> 7;
}
#include "main.inc"
