/** KDG two phase executor -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */

#ifndef GALOIS_RUNTIME_KDGTWOPHASE_H
#define GALOIS_RUNTIME_KDGTWOPHASE_H

#include "galois/GaloisForwardDecl.h"
#include "galois/Reduction.h"
#include "galois/Atomic.h"
#include "galois/BoundedVector.h"
#include "galois/gdeque.h"
#include "galois/PriorityQueue.h"
#include "galois/Timer.h"
#include "galois/DoAllWrap.h"
#include "galois/PerThreadContainer.h"
#include "galois/optional.h"

#include "galois/substrate/Barrier.h"
#include "galois/runtime/Context.h"
#include "galois/runtime/Executor_DoAll.h"
#include "galois/runtime/Executor_ParaMeter.h"
#include "galois/runtime/ForEachTraits.h"
#include "galois/runtime/Range.h"
#include "galois/runtime/Support.h"
#include "galois/substrate/Termination.h"
#include "galois/substrate/ThreadPool.h"
#include "galois/runtime/IKDGbase.h"
#include "galois/runtime/WindowWorkList.h"
#include "galois/runtime/UserContextAccess.h"
#include "galois/gIO.h"
#include "galois/substrate/CompilerSpecific.h"
#include "galois/runtime/Mem.h"

#include <boost/iterator/transform_iterator.hpp>

#include <iostream>
#include <memory>


namespace galois {
namespace runtime {


namespace {

template <typename T, typename Cmp, typename NhFunc, typename ExFunc, typename OpFunc, typename ArgsTuple>
class IKDGtwoPhaseExecutor: public IKDGbase<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple, TwoPhaseContext<T, Cmp> > {

  using ThisClass = IKDGtwoPhaseExecutor;

public:
  using Ctxt = TwoPhaseContext<T, Cmp>;
  using Base = IKDGbase <T, Cmp, NhFunc, ExFunc, OpFunc, ArgsTuple, Ctxt>;

  using CtxtWL = typename ThisClass::CtxtWL;



protected:

  static const bool DETAILED_STATS = false;

  struct CtxtMaker {
    IKDGtwoPhaseExecutor& outer;

    Ctxt* operator () (const T& x) {

      Ctxt* ctxt = outer.ctxtAlloc.allocate (1);
      assert (ctxt);
      outer.ctxtAlloc.construct (ctxt, x, outer.cmp);

      return ctxt;
    }
  };


  typename ThisClass::template WindowWLwrapper<IKDGtwoPhaseExecutor> winWL;
  CtxtMaker ctxtMaker;


public:
  IKDGtwoPhaseExecutor (
      const Cmp& cmp,
      const NhFunc& nhFunc,
      const ExFunc& exFunc,
      const OpFunc& opFunc,
      const ArgsTuple& argsTuple)
    :
      Base (cmp, nhFunc, exFunc, opFunc, argsTuple),
      winWL (*this, cmp),
      ctxtMaker {*this}
  {
  }

  ~IKDGtwoPhaseExecutor () {

    dumpStats ();

    if (ThisClass::ENABLE_PARAMETER) {
      ParaMeter::closeStatsFile ();
    }
  }

  void dumpStats (void) {
    reportStat_Single (Base::loopname, "efficiency %", double (100.0 * ThisClass::totalCommits) / ThisClass::totalTasks);
    reportStat_Single (Base::loopname, "avg. parallelism", double (ThisClass::totalCommits) / ThisClass::rounds);
  }

  CtxtMaker& getCtxtMaker(void) {
    return ctxtMaker;
  }

  template <typename R>
  void push_initial (const R& range) {
    if (ThisClass::targetCommitRatio == 0.0) {

      galois::runtime::do_all_gen (range,
          [this] (const T& x) {
            ThisClass::getNextWL ().push_back (ctxtMaker (x));
          },
          std::make_tuple (
            galois::loopname ("init-fill"),
            chunk_size<NhFunc::CHUNK_SIZE> ()));


    } else {
      winWL.initfill (range);

    }
  }

  void execute () {
    execute_impl ();
  }

protected:

  GALOIS_ATTRIBUTE_PROF_NOINLINE void endRound () {

    if (ThisClass::ENABLE_PARAMETER) {
      ParaMeter::StepStats s (ThisClass::rounds, ThisClass::roundCommits.reduceRO (), ThisClass::roundTasks.reduceRO ());
      s.dump (ParaMeter::getStatsFile (), ThisClass::loopname);
    }

    ThisClass::endRound ();
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void expandNhoodImpl (internal::DummyExecFunc*) {
    // for stable case

    galois::runtime::do_all_gen (makeLocalRange (ThisClass::getCurrWL ()),
        [this] (Ctxt* c) {
          typename ThisClass::UserCtxt& uhand = *ThisClass::userHandles.getLocal ();
          uhand.reset ();

          // nhFunc (c, uhand);
          runCatching (ThisClass::nhFunc, c, uhand);

          ThisClass::roundTasks += 1;
        },
        std::make_tuple (
          galois::loopname ("expandNhood"),
          chunk_size<NhFunc::CHUNK_SIZE> ()));
  }

  struct GetActive: public std::unary_function<Ctxt*, const T&> {
    const T& operator () (const Ctxt* c) const {
      assert (c != nullptr);
      return c->getActive ();
    }
  };

  template <typename F>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void expandNhoodImpl (F*) {
    // for unstable case
    auto m_beg = boost::make_transform_iterator (ThisClass::getCurrWL ().begin_all (), GetActive ());
    auto m_end = boost::make_transform_iterator (ThisClass::getCurrWL ().end_all (), GetActive ());

    galois::runtime::do_all_gen (makeLocalRange (ThisClass::getCurrWL ()),
        [m_beg, m_end, this] (Ctxt* c) {
          typename ThisClass::UserCtxt& uhand = *ThisClass::userHandles.getLocal ();
          uhand.reset ();

          runCatching (ThisClass::nhFunc, c, uhand, m_beg, m_end);

          ThisClass::roundTasks += 1;
        },
        std::make_tuple (
          galois::loopname ("expandNhoodUnstable"),
          chunk_size<NhFunc::CHUNK_SIZE> ()));
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void expandNhood () {
    // using ptr to exFunc to choose the right impl.
    // relying on the fact that for stable case, the exFunc is DummyExecFunc.
    expandNhoodImpl (&this->exFunc);
  }

  inline void executeSourcesImpl (internal::DummyExecFunc*) {
  }

  template <typename F>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void executeSourcesImpl (F*) {
    assert (ThisClass::HAS_EXEC_FUNC);

    galois::runtime::do_all_gen (makeLocalRange (ThisClass::getCurrWL ()),
      [this] (Ctxt* ctxt) {

        typename ThisClass::UserCtxt& uhand = *ThisClass::userHandles.getLocal ();
        uhand.reset ();

        if (ctxt->isSrc ()) {
          this->exFunc (ctxt->getActive (), uhand);
        }
      },
      std::make_tuple (
        galois::loopname ("exec-sources"),
        galois::chunk_size<ExFunc::CHUNK_SIZE> ()));

  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void executeSources (void) {
    // using ptr to exFunc to choose the right impl.
    // relying on the fact that for stable case, the exFunc is DummyExecFunc.
    executeSourcesImpl (&this->exFunc);
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void applyOperator () {
    galois::optional<T> minElem;

    if (ThisClass::NEEDS_PUSH) {
      if (ThisClass::targetCommitRatio != 0.0 && !winWL.empty ()) {
        minElem = *winWL.getMin();
      }
    }


    galois::runtime::do_all_gen (makeLocalRange (ThisClass::getCurrWL ()),
        [this, &minElem] (Ctxt* c) {
          bool commit = false;

          typename ThisClass::UserCtxt& uhand = *ThisClass::userHandles.getLocal ();
          uhand.reset ();

          if (ThisClass::NEEDS_CUSTOM_LOCKING || c->isSrc ()) {
            // opFunc (c->active, uhand);
            if (ThisClass::NEEDS_CUSTOM_LOCKING) {
              c->enableSrc();
              runCatching (ThisClass::opFunc, c, uhand);
              commit = c->isSrc (); // in case opFunc signalled abort

            } else {
              ThisClass::opFunc (c->getActive (), uhand);
              assert (c->isSrc ());
              commit = true;
            }
          } else {
            commit = false;
          }

          if (commit) {
            ThisClass::roundCommits += 1;
            if (ThisClass::NEEDS_PUSH) {
              for (auto i = uhand.getPushBuffer ().begin ()
                  , endi = uhand.getPushBuffer ().end (); i != endi; ++i) {

                if ((ThisClass::targetCommitRatio == 0.0) || !minElem || !ThisClass::cmp (*minElem, *i)) {
                  // if *i >= *minElem
                  ThisClass::getNextWL ().push_back (ctxtMaker (*i));
                } else {
                  winWL.push (*i);
                }
              }
            } else {
              assert (uhand.getPushBuffer ().begin () == uhand.getPushBuffer ().end ());
            }

            c->commitIteration ();
            c->~Ctxt ();
            ThisClass::ctxtAlloc.deallocate (c, 1);
          } else {
            c->cancelIteration ();
            c->reset ();
            ThisClass::getNextWL ().push_back (c);
          }
        },
        std::make_tuple (
          galois::loopname ("applyOperator"),
          chunk_size<OpFunc::CHUNK_SIZE> ()));
  }


  void execute_impl () {

    StatTimer t ("executorLoop");
    t.start();

    while (true) {
      ThisClass::t_beginRound.start();
      ThisClass::beginRound (winWL);

      if (ThisClass::getCurrWL ().empty_all ()) {
        break;
      }
      ThisClass::t_beginRound.stop();

      Timer t;

      ThisClass::t_expandNhood.start();
      expandNhood ();
      ThisClass::t_expandNhood.stop();

      ThisClass::t_executeSources.start();
      executeSources ();
      ThisClass::t_executeSources.stop();

      ThisClass::t_applyOperator.start();
      applyOperator ();
      ThisClass::t_applyOperator.stop();

      endRound ();

    }

    t.stop();
  }

};


} // end anonymous namespace

template <typename R, typename Cmp, typename NhFunc, typename ExFunc, typename OpFunc, typename _ArgsTuple>
void for_each_ordered_ikdg_impl (const R& range, const Cmp& cmp, const NhFunc& nhFunc,
    const ExFunc& exFunc,  const OpFunc& opFunc, const _ArgsTuple& argsTuple) {

  auto argsT = std::tuple_cat (argsTuple,
      get_default_trait_values (argsTuple,
        std::make_tuple (loopname_tag {}, enable_parameter_tag {}),
        std::make_tuple (default_loopname {}, enable_parameter<false> {})));
  using ArgsT = decltype (argsT);

  using T = typename R::value_type;


  using Exec = IKDGtwoPhaseExecutor<T, Cmp, NhFunc, ExFunc, OpFunc, ArgsT>;

  Exec e (cmp, nhFunc, exFunc, opFunc, argsT);

  const bool wakeupThreadPool = true;

  if (wakeupThreadPool) {
    substrate::getThreadPool().burnPower(galois::getActiveThreads ());
  }

  e.push_initial (range);
  e.execute ();

  if (wakeupThreadPool) {
    substrate::getThreadPool().beKind ();
  }

}

template <typename R, typename Cmp, typename NhFunc, typename ExFunc, typename OpFunc, typename _ArgsTuple>
void for_each_ordered_ikdg (const R& range, const Cmp& cmp, const NhFunc& nhFunc,
    const ExFunc& exFunc,  const OpFunc& opFunc, const _ArgsTuple& argsTuple) {

  auto tplParam = std::tuple_cat (argsTuple, std::make_tuple (enable_parameter<true> ()));
  auto tplNoParam = std::tuple_cat (argsTuple, std::make_tuple (enable_parameter<false> ()));

  if (useParaMeterOpt) {
    for_each_ordered_ikdg_impl (range, cmp, nhFunc, exFunc, opFunc, tplParam);
  } else {
    for_each_ordered_ikdg_impl (range, cmp, nhFunc, exFunc, opFunc, tplNoParam);
  }
}

template <typename R, typename Cmp, typename NhFunc, typename OpFunc, typename ArgsTuple>
void for_each_ordered_ikdg (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const ArgsTuple& argsTuple) {

  for_each_ordered_ikdg (range, cmp, nhFunc, internal::DummyExecFunc (), opFunc, argsTuple);
}

} // end namespace runtime
} // end namespace galois

#endif //  GALOIS_RUNTIME_KDG_TWO_PHASE_H
