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

template <typename T, typename Cmp, typename NhoodFunc, typename OpFunc, typename G>
struct DetKDGexecutor {

  typedef Galois::PerThreadBag<T> Bag_ty;
  // typedef Galois::Runtime::PerThreadVector<T> Bag_ty;

  static const unsigned DEFAULT_CHUNK_SIZE = 8;

  Cmp cmp;
  NhoodFunc nhoodVisitor;
  OpFunc opFunc;
  G& graph; 
  const char* loopname;
  unsigned rounds = 0; 
  

  Bag_ty* currWL;
  Bag_ty* nextWL;
  

  DetKDGexecutor (
      const Cmp& cmp,
      const NhoodFunc& nhoodVisitor, 
      const OpFunc& opFunc, 
      G& graph, 
      const char* loopname)
    :
      cmp (cmp),
      nhoodVisitor (nhoodVisitor),
      opFunc (opFunc),
      graph (graph),
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

    auto& elemData = graph.getData (elem, Galois::MethodFlag::UNPROTECTED);

    unsigned expected = rounds;
    const unsigned update = rounds + 1;
    if (elemData.onWL.compare_exchange_strong (expected, update)) {
      nextWL->get ().push_back (elem);
    }
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

    rounds = 0;
    Galois::Timer t_exec;
    while (!nextWL->empty_all ()) {
      ++rounds;
      std::swap (currWL, nextWL);
      nextWL->clear_all_parallel ();

      t_exec.start ();
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
      t_exec.stop ();

      if (rounds >= 2) { 
        // break; // TODO: remove
      }
      std::printf ("DetKDGreuseDAGexec: round %d time taken: %ld\n", rounds, t_exec.get ());

    } // end while

    std::printf ("DetKDGexecutor: performed %d rounds\n", rounds);

  }


};


template <typename R, typename Cmp, typename NhoodFunc, typename OpFunc, typename G>
void for_each_det_kdg (const R& initRange, const Cmp& cmp, const NhoodFunc& nhoodVisitor, 
    const OpFunc& opFunc, G& graph, const char* loopname, const KDGexecType& kdgType) {

  Galois::Runtime::getSystemThreadPool ().burnPower (Galois::getActiveThreads ());

  typedef typename R::value_type T;

  DetKDGexecutor<T, Cmp, NhoodFunc, OpFunc, G> executor {cmp, nhoodVisitor, opFunc, graph, loopname};

  executor.execute (initRange, kdgType);

  Galois::Runtime::getSystemThreadPool ().beKind ();
}

template <typename T, typename Cmp, typename NhoodFunc, typename OpFunc, typename G>
struct DetKDGreuseDAGexec {

  typedef Galois::PerThreadBag<T> Bag_ty;
  // typedef Galois::Runtime::PerThreadVector<T> Bag_ty;

  static const unsigned DEFAULT_CHUNK_SIZE = 8;

  Cmp cmp;
  NhoodFunc nhoodVisitor;
  OpFunc opFunc;
  G& graph; 
  const char* loopname;
  unsigned rounds = 0; 
  Galois::GAccumulator<size_t> numPushes;
  

  DetKDGreuseDAGexec (
      const Cmp& cmp,
      const NhoodFunc& nhoodVisitor, 
      const OpFunc& opFunc, 
      G& graph, 
      const char* loopname)
    :
      cmp (cmp),
      nhoodVisitor (nhoodVisitor),
      opFunc (opFunc),
      graph (graph),
      loopname (loopname)
  {
  }

  struct ApplyOperator {
    static const unsigned CHUNK_SIZE = OpFunc::CHUNK_SIZE;
    static const unsigned UNROLL_FACTOR = OpFunc::UNROLL_FACTOR;

    typedef int tt_does_not_need_push;

    DetKDGreuseDAGexec& outer;

    template <typename C>
    void operator () (T elem, C& ctx) {
      auto& edata = outer.graph.getData (elem, Galois::MethodFlag::UNPROTECTED);

      if (edata.onWL > 0) {
        outer.opFunc (elem, outer);
        --(edata.onWL);
      }
    }
  };

  void push (const T& elem) {
    numPushes += 1;
    auto& edata = graph.getData (elem, Galois::MethodFlag::UNPROTECTED);
    ++(edata.onWL);
  }

  template <typename R>
  void execute (const R& initRange) {

    Galois::do_all_choice (
        initRange,
        [this] (T node) {
          push (node);
        }, 
        "push_initial",
        Galois::doall_chunk_size<DEFAULT_CHUNK_SIZE> ());

    auto* dagExec = make_dag_executor (initRange, cmp, nhoodVisitor, ApplyOperator{*this}, loopname);

    dagExec->initialize(initRange);


    Galois::Timer t_exec;
    rounds = 0;
    while (true) {
      ++rounds;

      t_exec.start ();
      dagExec->execute ();
      t_exec.stop ();

      if (numPushes.reduceRO () == 0) { 
        break;
      }
      std::printf ("DetKDGreuseDAGexec: round %d time taken: %ld\n", rounds, t_exec.get ());

      dagExec->resetDAG ();
      numPushes.reset ();
    }

    // destroy_dag_executor (dagExec);
    delete dagExec; dagExec = nullptr;

    std::printf ("DetKDGreuseDAGexec: performed %d rounds\n", rounds);
  }


};


template <typename R, typename Cmp, typename NhoodFunc, typename OpFunc, typename G>
void for_each_det_kdg_topo (const R& initRange, const Cmp& cmp, const NhoodFunc& nhoodVisitor, 
    const OpFunc& opFunc, G& graph, const char* loopname) {

  Galois::Runtime::getSystemThreadPool ().burnPower (Galois::getActiveThreads ());

  typedef typename R::value_type T;

  DetKDGreuseDAGexec<T, Cmp, NhoodFunc, OpFunc, G> executor {cmp, nhoodVisitor, opFunc, graph, loopname};

  executor.execute (initRange);

  Galois::Runtime::getSystemThreadPool ().beKind ();
}

} //end namespace Runtime
} // end namespace Galois

#endif // GALOIS_RUNTIME_DET_KDG_EXECUTOR_H
