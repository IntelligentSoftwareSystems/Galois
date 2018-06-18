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

#ifndef GALOIS_RUNTIME_DET_KDG_EXECUTOR_H
#define GALOIS_RUNTIME_DET_KDG_EXECUTOR_H

#include "galois/Atomic.h"

#include "galois/runtime/DetChromatic.h"
#include "galois/runtime/LCordered.h"
#include "galois/runtime/KDGtwoPhase.h"
#include "galois/runtime/DAGexec.h"
#include "galois/runtime/DAGexecAlt.h"

namespace galois {
namespace runtime {

enum KDGexecType { KDG_R_ALT, KDG_R, KDG_AR, IKDG };

template <typename T, typename Cmp, typename NhoodFunc, typename OpFunc,
          typename G>
struct DetKDGexecutorAddRem {

  typedef galois::PerThreadBag<T> Bag_ty;
  // typedef galois::runtime::PerThreadVector<T> Bag_ty;

  static const unsigned DEFAULT_CHUNK_SIZE = 8;

  Cmp cmp;
  NhoodFunc nhoodVisitor;
  OpFunc opFunc;
  G& graph;
  const char* loopname;
  unsigned rounds = 0;

  Bag_ty* currWL;
  Bag_ty* nextWL;

  DetKDGexecutorAddRem(const Cmp& cmp, const NhoodFunc& nhoodVisitor,
                       const OpFunc& opFunc, G& graph, const char* loopname)
      : cmp(cmp), nhoodVisitor(nhoodVisitor), opFunc(opFunc), graph(graph),
        loopname(loopname) {
    currWL = new Bag_ty();
    nextWL = new Bag_ty();
  }

  ~DetKDGexecutorAddRem(void) {
    delete currWL;
    currWL = nullptr;
    delete nextWL;
    nextWL = nullptr;
  }

  void push(const T& elem) {

    auto& elemData = graph.getData(elem, galois::MethodFlag::UNPROTECTED);

    unsigned expected     = rounds;
    const unsigned update = rounds + 1;
    if (elemData.onWL.cas(expected, update)) {
      nextWL->get().push_back(elem);
    }
  }

  struct ApplyOperator {

    static const unsigned CHUNK_SIZE    = OpFunc::CHUNK_SIZE;
    static const unsigned UNROLL_FACTOR = OpFunc::UNROLL_FACTOR;

    typedef int tt_does_not_need_push;

    DetKDGexecutorAddRem& outer;

    template <typename C>
    void operator()(const T& elem, C& ctx) {
      outer.opFunc(elem, outer);
    }
  };

  template <typename R>
  void execute(const R& range, KDGexecType kdgType) {

    galois::runtime::do_all_gen(range, [this](const T& elem) { push(elem); },
                                "push_initial",
                                galois::chunk_size<DEFAULT_CHUNK_SIZE>());

    rounds = 0;
    galois::Timer t_exec;
    while (!nextWL->empty_all()) {
      ++rounds;
      std::swap(currWL, nextWL);
      nextWL->clear_all_parallel();

      t_exec.start();
      switch (kdgType) {
      case KDG_R_ALT:
        for_each_ordered_dag_alt(galois::runtime::makeLocalRange(*currWL), cmp,
                                 nhoodVisitor, ApplyOperator{*this}, "kdg_r");
        break;

      case KDG_R:
        for_each_ordered_dag(galois::runtime::makeLocalRange(*currWL), cmp,
                             nhoodVisitor, ApplyOperator{*this}, "kdg_r");
        break;

      case KDG_AR:
        for_each_ordered_lc(galois::runtime::makeLocalRange(*currWL), cmp,
                            nhoodVisitor, ApplyOperator{*this}, "kdg_ar");
        break;

        // case IKDG:
        //  for_each_ordered_2p_win (
        //      galois::runtime::makeLocalRange (*currWL),
        //      cmp,
        //      nhoodVisitor,
        //      ApplyOperator{*this},
        //      "ikdg", false); // false to avoid toggling threadPool wakeup
        //  break;

      default:
        std::abort();

      } // end switch
      t_exec.stop();

      if (rounds >= 2) {
        // break; // TODO: remove
      }
      // std::printf ("DetKDGexecutorAddRem round %d time taken: %ld\n", rounds,
      // t_exec.get ());

    } // end while

    std::printf("DetKDGexecutorAddRem: performed %d rounds\n", rounds);
  }
};

template <typename R, typename Cmp, typename NhoodFunc, typename OpFunc,
          typename G>
void for_each_det_kdg(const R& initRange, const Cmp& cmp,
                      const NhoodFunc& nhoodVisitor, const OpFunc& opFunc,
                      G& graph, const char* loopname,
                      const KDGexecType& kdgType) {

  galois::substrate::getThreadPool().burnPower(galois::getActiveThreads());

  typedef typename R::value_type T;

  DetKDGexecutorAddRem<T, Cmp, NhoodFunc, OpFunc, G> executor{
      cmp, nhoodVisitor, opFunc, graph, loopname};

  executor.execute(initRange, kdgType);

  galois::substrate::getThreadPool().beKind();
}

template <typename T, typename Cmp, typename NhoodFunc, typename OpFunc,
          typename G>
struct DetKDG_AddRem_reuseDAG {

  typedef galois::PerThreadBag<T> Bag_ty;
  // typedef galois::runtime::PerThreadVector<T> Bag_ty;

  static const unsigned DEFAULT_CHUNK_SIZE = 8;

  Cmp cmp;
  NhoodFunc nhoodVisitor;
  OpFunc opFunc;
  G& graph;
  const char* loopname;
  unsigned rounds = 0;
  galois::GAccumulator<size_t> numPushes;

  DetKDG_AddRem_reuseDAG(const Cmp& cmp, const NhoodFunc& nhoodVisitor,
                         const OpFunc& opFunc, G& graph, const char* loopname)
      : cmp(cmp), nhoodVisitor(nhoodVisitor), opFunc(opFunc), graph(graph),
        loopname(loopname) {}

  struct ApplyOperator {
    static const unsigned CHUNK_SIZE    = OpFunc::CHUNK_SIZE;
    static const unsigned UNROLL_FACTOR = OpFunc::UNROLL_FACTOR;

    typedef int tt_does_not_need_push;

    DetKDG_AddRem_reuseDAG& outer;

    template <typename C>
    void operator()(T elem, C& ctx) {
      auto& edata = outer.graph.getData(elem, galois::MethodFlag::UNPROTECTED);

      if (edata.onWL > 0) {
        outer.opFunc(elem, outer);
        --(edata.onWL);
      }
    }
  };

  void push(const T& elem) {
    numPushes += 1;
    auto& edata = graph.getData(elem, galois::MethodFlag::UNPROTECTED);
    ++(edata.onWL);
  }

  template <typename R>
  void execute(const R& initRange) {

    galois::runtime::do_all_gen(initRange, [this](T node) { push(node); },
                                "push_initial",
                                galois::chunk_size<DEFAULT_CHUNK_SIZE>());

    auto* dagExec = make_dag_executor(initRange, cmp, nhoodVisitor,
                                      ApplyOperator{*this}, loopname);

    dagExec->initialize(initRange);

    galois::Timer t_exec;
    rounds = 0;
    while (true) {
      ++rounds;

      t_exec.start();
      dagExec->execute();
      t_exec.stop();

      if (numPushes.reduceRO() == 0) {
        break;
      }
      // std::printf ("DetKDG_AddRem_reuseDAG: round %d time taken: %ld\n",
      // rounds, t_exec.get ());

      abort();
      // FIXME:      dagExec->reinitDAG ();
      numPushes.reset();
    }

    // destroy_dag_executor (dagExec);
    delete dagExec;
    dagExec = nullptr;

    std::printf("DetKDG_AddRem_reuseDAG: performed %d rounds\n", rounds);
  }
};

template <typename R, typename Cmp, typename NhoodFunc, typename OpFunc,
          typename G>
void for_each_det_kdg_ar_reuse(const R& initRange, const Cmp& cmp,
                               const NhoodFunc& nhoodVisitor,
                               const OpFunc& opFunc, G& graph,
                               const char* loopname) {

  galois::substrate::getThreadPool().burnPower(galois::getActiveThreads());

  typedef typename R::value_type T;

  DetKDG_AddRem_reuseDAG<T, Cmp, NhoodFunc, OpFunc, G> executor{
      cmp, nhoodVisitor, opFunc, graph, loopname};

  executor.execute(initRange);

  galois::substrate::getThreadPool().beKind();
}

} // end namespace runtime
} // end namespace galois

#endif // GALOIS_RUNTIME_DET_KDG_EXECUTOR_H
