#ifndef GALOIS_DET_SCHEDULES_H
#define GALOIS_DET_SCHEDULES_H

#include "Galois/Runtime/Executor_Deterministic.h"
#include "Galois/Runtime/DAGexec.h"
#include "Galois/Runtime/DetChromatic.h"
#include "Galois/Runtime/DetKDGexecutor.h"

#include "llvm/Support/CommandLine.h"

namespace Galois {

enum DetExecType {
  non_det,
  det_i,
  det_ar,
  kdg_i,
  kdg_ar,
  chromatic,
  topo,
  edge_flip
};

static cll::opt<DetExecType> detExecTypeArg("detexec", cll::desc("Choose schedule for asynchronous algorithm"),
  cll::values(
    clEnumValN(non_det, "non_det", "non deterministic using for_each"),
    clEnumValN(det_i, "det_i", "deterministic using implicit kdg"),
    clEnumValN(det_ar, "det_ar", "deterministic add-remove"),
    clEnumValN(chromatic, "chromatic", "chromatic"),
    clEnumValN(topo, "topo", "topo"),
    clEnumValN(edge_flip, "edge_flip", "edge_flip"),
  clEnumValEnd),
  cll::init(det_i));


#if 0
template <typename R, typename F>
void for_each_det_choice (const R& range, const F& func, const char* loopname, const DetExecType& detExec=detExecTypeArg) {

  switch (detExec) {
    case non_det:
      {
        const unsigned CHUNK_SIZE = 32;
        typedef Galois::WorkList::dChunkedFIFO<CHUNK_SIZE, typename R::value_type> WL_ty;
        Galois::Runtime::for_each_impl<WL_ty> (range, func, loopname);
        break;
      }

    case det_i:
      Galois::Runtime::for_each_det_impl (range, func, loopname);
      break;

    case det_ar:
      Galois::Runtime::for_each_det_impl (range, func, loopname);
      break;

    default:
      GALOIS_DIE ("not implemented");
      break;
  }
}
#endif

template <typename R, typename C, typename F, typename N>
void for_each_det_choice (const R&  range, const C& cmp, const N& nhoodVisitor, const F& func, const char* loopname, const DetExecType& detExec=detExecTypeArg) {

  switch (detExec) {
    case kdg_i: 
      Galois::Runtime::for_each_ordered_2p_win (range, cmp, nhoodVisitor, func, loopname);
      break;

    case kdg_ar:
      GALOIS_DIE ("not implemented yet");
      break;

    default:
      GALOIS_DIE ("not implemented");
      break;

  }

}

template <typename R, typename F, typename G>
void for_each_det_choice (const R& range, const F& func, G& graph, const char* loopname, const DetExecType& detExec=detExecTypeArg) {

  switch (detExec) {
    case chromatic:
      Galois::Runtime::for_each_det_chromatic (range, func, graph, loopname);
      break;

    case edge_flip:
      Galois::Runtime::for_each_det_edge_flip_ar (range, func, graph, loopname);
      break;

    case topo:
      Galois::Runtime::for_each_det_edge_flip_topo (range, func, graph, loopname);
      break;

    default:
      GALOIS_DIE ("not implemented");
      break;
  }
}

template <typename R, typename F, typename G, typename M>
void for_each_det_choice (const R& range, const F& func, G& graph, M& dagManager, const char* loopname, const DetExecType& detExec=detExecTypeArg) {

  switch (detExec) {
    case chromatic:
      Galois::Runtime::for_each_det_chromatic (range, func, graph, dagManager, loopname);
      break;

    case edge_flip:
      Galois::Runtime::for_each_det_edge_flip_ar (range, func, graph, dagManager, loopname);
      break;

    case topo:
      Galois::Runtime::for_each_det_edge_flip_topo (range, func, graph, dagManager, loopname);
      break;

    default:
      GALOIS_DIE ("not implemented");
      break;

  }

}





} // end namespace Galois


#endif //  GALOIS_DET_SCHEDULES_H
