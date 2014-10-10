#ifndef GALOIS_RUNTIME_DET_KDG_EXECUTOR_H
#define GALOIS_RUNTIME_DET_KDG_EXECUTOR_H

#include "Galois/Runtime/LCordered.h"
#include "Galois/Runtime/KDGtwoPhase.h"
#include "Galois/Runtime/DAGexec.h"
#include "Galois/Runtime/DAGexecAlt.h"

namespace Galois {
namespace Runtime {

enum KDGexecType {
  KDG_R_ALT,
  KDG_R,
  KDG_AR,
  IKDG
};

template <typename T, typename Cmp, typename NhoodFunc, typename OpFunc>
struct DetKDGexecutor {

  // typedef Galois::PerThreadBag<T> Bag_ty;
  typedef Galois::Runtime::PerThreadVector<T> Bag_ty;

  static const unsigned DEFAULT_CHUNK_SIZE = 8;

  Cmp cmp;
  NhoodFunc nhoodVisitor;
  OpFunc opFunc;
  const char* loopname;

  Bag_ty* currWL;
  Bag_ty* nextWL;

  DetKDGexecutor (
      const Cmp& cmp,
      const NhoodFunc& nhoodVisitor, 
      const OpFunc& opFunc, 
      const char* loopname)
    :
      cmp (cmp),
      nhoodVisitor (nhoodVisitor),
      opFunc (opFunc),
      loopname (loopname)
  {
    currWL = new Bag_ty ();
    nextWL = new Bag_ty ();
  }

  ~DetKDGexecutor (void) {
    delete currWL; currWL = nullptr;
    delete nextWL; nextWL = nullptr;
  }


  void push (const T& elem) {
    nextWL->get ().push_back (elem);
  }

  struct ApplyOperator {

    static const unsigned CHUNK_SIZE = OpFunc::CHUNK_SIZE;
    static const unsigned UNROLL_FACTOR = OpFunc::UNROLL_FACTOR;

    typedef int tt_does_not_need_push;

    DetKDGexecutor& outer;

    template <typename C>
    void operator () (const T& elem, C& ctx) {
      outer.opFunc (elem, outer);
    }
  };

  template <typename R>
  void execute (const R& range, KDGexecType kdgType) {

    Galois::do_all_choice (
        range,
        [this] (const T& elem) {
          push (elem);
        }, 
        "push_initial",
        Galois::doall_chunk_size<DEFAULT_CHUNK_SIZE> ());

    unsigned rounds = 0;
    while (!nextWL->empty_all ()) {
      ++rounds;
      std::swap (currWL, nextWL);
      nextWL->clear_all_parallel ();

      switch (kdgType) {
        case KDG_R_ALT:
          for_each_ordered_dag_alt (
              Galois::Runtime::makeLocalRange (*currWL),
              cmp,
              nhoodVisitor,
              ApplyOperator{*this},
              "kdg_r");
          break;

        case KDG_R:
          for_each_ordered_dag (
              Galois::Runtime::makeLocalRange (*currWL),
              cmp,
              nhoodVisitor,
              ApplyOperator{*this},
              "kdg_r");
          break;

        case KDG_AR:
          for_each_ordered_lc (
              Galois::Runtime::makeLocalRange (*currWL),
              cmp,
              nhoodVisitor,
              ApplyOperator{*this},
              "kdg_ar");
          break;

        case IKDG:
          for_each_ordered_2p_win (
              Galois::Runtime::makeLocalRange (*currWL),
              cmp,
              nhoodVisitor,
              ApplyOperator{*this},
              "ikdg", false); // false to avoid toggling threadPool wakeup
          break;

        default:
          std::abort ();

      } // end switch

    } // end while

    std::printf ("DetKDGexecutor: performed %d rounds\n", rounds);

  }


};


template <typename R, typename Cmp, typename NhoodFunc, typename OpFunc>
void for_each_det_kdg (const R& initRange, const Cmp& cmp, const NhoodFunc& nhoodVisitor, const OpFunc& opFunc, const char* loopname, const KDGexecType& kdgType) {

  Galois::Runtime::getSystemThreadPool ().burnPower (Galois::getActiveThreads ());

  typedef typename R::value_type T;

  DetKDGexecutor<T, Cmp, NhoodFunc, OpFunc> executor {cmp, nhoodVisitor, opFunc, loopname};

  executor.execute (initRange, kdgType);

  Galois::Runtime::getSystemThreadPool ().beKind ();
}

} //end namespace Runtime
} // end namespace Galois

#endif // GALOIS_RUNTIME_DET_KDG_EXECUTOR_H
