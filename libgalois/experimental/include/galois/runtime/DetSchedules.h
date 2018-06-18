/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
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

#ifndef GALOIS_DET_SCHEDULES_H
#define GALOIS_DET_SCHEDULES_H

#include "galois/runtime/Executor_Deterministic.h"
#include "galois/runtime/DAGexec.h"
#include "galois/runtime/DetChromatic.h"
#include "galois/runtime/DetKDGexecutor.h"

#include "llvm/Support/CommandLine.h"

namespace galois {

namespace runtime {

enum DetExecType {
  non_det,
  det_i,
  det_ar,
  kdg_i,
  kdg_ar,
  kdg_r,
  chromatic,
  topo,
  edge_flip
};

static cll::opt<DetExecType> detExecTypeArg(
    "detexec", cll::desc("Choose schedule for asynchronous algorithm"),
    cll::values(clEnumValN(non_det, "non_det",
                           "non deterministic using for_each"),
                clEnumValN(det_i, "det_i", "deterministic using implicit kdg"),
                clEnumValN(det_ar, "det_ar", "deterministic add-remove"),
                clEnumValN(kdg_r, "kdg_r", "kdg_r"),
                clEnumValN(chromatic, "chromatic", "chromatic"),
                clEnumValN(topo, "topo", "topo"),
                clEnumValN(edge_flip, "edge_flip", "edge_flip"), clEnumValEnd),
    cll::init(det_i));

#if 0
template <typename R, typename F>
void for_each_det_choice (const R& range, const F& func, const char* loopname, const DetExecType& detExec=detExecTypeArg) {

  switch (detExec) {
    case non_det:
      {
        const unsigned CHUNK_SIZE = 32;
        typedef galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE, typename R::value_type> WL_ty;
        galois::runtime::for_each_impl<WL_ty> (range, func, loopname);
        break;
      }

    case det_i:
      galois::runtime::for_each_det_impl (range, func, loopname);
      break;

    case det_ar:
      galois::runtime::for_each_det_impl (range, func, loopname);
      break;

    default:
      GALOIS_DIE ("not implemented");
      break;
  }
}
#endif

template <typename R, typename C, typename F, typename N>
void for_each_det_choice(const R& range, const C& cmp, const N& nhoodVisitor,
                         const F& func, const char* loopname,
                         const DetExecType& detExec = detExecTypeArg) {

  switch (detExec) {
    // case kdg_i:
    //  galois::runtime::for_each_ordered_2p_win (range, cmp, nhoodVisitor,
    //  func, loopname); break;

  case kdg_ar:
    GALOIS_DIE("not implemented yet");
    break;

  default:
    GALOIS_DIE("not implemented");
    break;
  }
}

template <typename R, typename F, typename G>
void for_each_det_choice(const R& range, const F& func, G& graph,
                         const char* loopname,
                         const DetExecType& detExec = detExecTypeArg) {

  switch (detExec) {
  case chromatic:
    galois::runtime::for_each_det_chromatic(range, func, graph, loopname);
    break;

  case edge_flip:
    galois::runtime::for_each_det_edge_flip_ar(range, func, graph, loopname);
    break;

  case topo:
    galois::runtime::for_each_det_edge_flip_topo(range, func, graph, loopname);
    break;

  default:
    GALOIS_DIE("not implemented");
    break;
  }
}

template <typename R, typename F, typename G, typename M>
void for_each_det_choice(const R& range, const F& func, G& graph, M& dagManager,
                         const char* loopname,
                         const DetExecType& detExec = detExecTypeArg) {

  switch (detExec) {
  case chromatic:
    galois::runtime::for_each_det_chromatic(range, func, graph, dagManager,
                                            loopname);
    break;

  case edge_flip:
    galois::runtime::for_each_det_edge_flip_ar(range, func, graph, dagManager,
                                               loopname);
    break;

  case topo:
    galois::runtime::for_each_det_edge_flip_topo(range, func, graph, dagManager,
                                                 loopname);
    break;

  default:
    GALOIS_DIE("not implemented");
    break;
  }
}

template <typename T, typename G, typename M, typename F, typename N,
          typename C>
struct ReuseableExecutorWrapper {

  typedef galois::runtime::ChromaticReuseExecutor<G, M, F> Chrom;
  typedef galois::runtime::InputGraphDAGreuseExecutor<G, M, F> InputDAG;
  typedef galois::runtime::DAGexecutorRW<T, C, F, N> TaskDAG;

  DetExecType detExec;

  Chrom chromExec;
  InputDAG inputDAGexec;
  TaskDAG taskDAGexec;

  ReuseableExecutorWrapper(DetExecType detExec, G& graph, M& dagManager,
                           const F& func, const N& nhVisitor, const C& cmp,
                           const char* loopname)
      : detExec{detExec}, chromExec{graph, dagManager, func, loopname},
        inputDAGexec{graph, dagManager, func, loopname}, taskDAGexec{
                                                             cmp, nhVisitor,
                                                             func, loopname} {}

  template <typename R>
  void initialize(const R& range) {

    switch (detExec) {

    case chromatic:
      chromExec.initialize(range);
      break;

    case edge_flip:
      inputDAGexec.initialize(range);
      break;

    case kdg_r:
      taskDAGexec.initialize(range);
      break;

    default:
      GALOIS_DIE("det exec type not supported");
      break;
    }
  }

  void execute(void) {
    switch (detExec) {

    case chromatic:
      chromExec.execute();
      break;

    case edge_flip:
      inputDAGexec.execute();
      break;

    case kdg_r:
      taskDAGexec.execute();
      break;

    default:
      GALOIS_DIE("det exec type not supported");
      break;
    }
  }

  void resetDAG(void) {
    switch (detExec) {

    case chromatic:
      chromExec.resetDAG();
      break;

    case edge_flip:
      inputDAGexec.resetDAG();
      break;

    case kdg_r:
      taskDAGexec.resetDAG();
      break;

    default:
      GALOIS_DIE("det exec type not supported");
      break;
    }
  }
};

template <typename R, typename G, typename M, typename F, typename N,
          typename C>

ReuseableExecutorWrapper<typename R::value_type, G, M, F, N, C>*
make_reusable_dag_exec(const R& range, G& graph, M& dagManager, const F& func,
                       const N& nhVisitor, const C& cmp, const char* loopname,
                       DetExecType detExec = detExecTypeArg) {

  return new ReuseableExecutorWrapper<typename R::value_type, G, M, F, N, C>{
      detExec, graph, dagManager, func, nhVisitor, cmp, loopname};
}

} // end namespace runtime

} // end namespace galois

#endif //  GALOIS_DET_SCHEDULES_H
