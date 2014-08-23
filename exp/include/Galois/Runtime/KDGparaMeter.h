/** KDG parameter based on two phase executor -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 *
 *
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_KDG_PARAMETER_H
#define GALOIS_RUNTIME_KDG_PARAMETER_H

#include "Galois/Accumulator.h"
#include "Galois/Atomic.h"
#include "Galois/BoundedVector.h"
#include "Galois/gdeque.h"
#include "Galois/PriorityQueue.h"
#include "Galois/Timer.h"

#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/DoAll.h"
#include "Galois/Runtime/ForEachTraits.h"
#include "Galois/Runtime/ParallelWork.h"
#include "Galois/Runtime/PerThreadContainer.h"
#include "Galois/Runtime/Range.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/KDGtwoPhaseSupport.h"
#include "Galois/Runtime/UserContextAccess.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/ll/ThreadRWlock.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"
#include "Galois/Runtime/mm/Mem.h"

#include <boost/iterator/transform_iterator.hpp>

#include <iostream>

// TODO: figure out when to call startIteration

namespace Galois {
namespace Runtime {

template <typename T, typename Cmp, typename NhFunc, typename OpFunc, typename ST>
class KDGtwoPhaseParaMeter {

  using Ctxt = TwoPhaseContext<T, Cmp>;
  using CtxtAlloc = MM::FixedSizeAllocator<Ctxt>;
  using CtxtWL = PerThreadVector<Ctxt*>;

  using UserCtx = UserContextAccess<T>;
  using PerThreadUserCtx = PerThreadStorage<UserCtx>;


  Cmp cmp;
  NhFunc nhFunc;
  OpFunc opFunc;
  SafetyTestLoop<ST> stloop;
  CtxtAlloc ctxtAlloc;
  PerThreadUserCtx userHandles;

  CtxtWL* currWL = new CtxtWL ();
  CtxtWL* nextWL = new CtxtWL ();

public:

  KDGtwoPhaseParaMeter (
      const Cmp& cmp, 
      const NhFunc& nhFunc,
      const OpFunc& opFunc,
      const ST& st)
    :
      cmp (cmp),
      nhFunc (nhFunc),
      opFunc (opFunc),
      stloop (st)
  {}

  ~KDGtwoPhaseParaMeter () {
    delete currWL; currWL = nullptr;
    delete nextWL; nextWL = nullptr;
  }



  template <typename I>
  void fill_initial (I beg, I end) {
    Galois::do_all (beg, end,
        [this] (const T& x) {
          createContext (x);
        },
        "fill_initial");
  }


  void execute () {

    size_t rounds = 0;
    GAccumulator<size_t> totalCommitted;

    while (!nextWL->empty_all ()) {

      ++rounds;
      std::swap (nextWL, currWL);
      // TODO: deallocate/destroy the allocated contexts
      nextWL->clear_all ();


      // expand nhood
      Galois::do_all (currWL->begin_all (), currWL->end_all (),
          [this] (Ctxt* c) {
            UserCtxt& uhand = *userHandles.getLocal ();
            uhand.reset ();

            // nhFunc (c, uhand);
            runCatching (nhFunc, c, uhand);
          },
          "expand_nhood");


      // collect sources
      // release locks here to save a separate phase
      allSources.clear_all ();
      Galois::do_all (currWL->begin_all (), currWL->end_all (),
          [this] (Ctxt* c) {
            if (c->isSrc ()) {
              allSources.get ().push_back (c);
              c->commitIteration ();

            } else {
              nextWL->get ().push_back (c);
              c->cancelIteration ();
            }
            c->reset (); // for future reuse
          },
          "collect_sources");

      // apply stability test

      auto iter_p = stloop.run (allSources.begin_all (), allSources.end_all ());

      // apply operator 
      Galois::do_all (iter_p.first, iter_p.second,
          [this, &numCommitted] (Ctxt* c) {
            assert (c->isSrc ());

            UserCtxt& uhand = *userHandles.getLocal ();
            // opFunc (c->active, uhand);
            runCatching (opFunc, c, uhand);

            // TODO: double check logic here
            if (c->isSrc ()) {

              numCommitted += 1;

              for (auto i = uhand.getPushBuffer ().begin ()
                , endi = uhand.getPushBuffer ().end (); i != endi; ++i) {
                
                createContext (*i);
              }

            } else {

              c->reset ();
              nextWL->get ().push_back (c);
            }

            uhand.reset ();
          },
          "apply_operator");

    } // end while

    std::cout << "KDGtwoPhaseParaMeter: number of rounds: " << rounds << std::endl;
    std::cout << "KDGtwoPhaseParaMeter: average parallelism: " << double (numCommitted.reduce ())/double(rounds) << std::endl;
  }

private:

  void createContext (const T& x) {
    Ctxt* ctx = ctxtAlloc.allocate (1);
    assert (ctx != nullptr);
    ctxtAlloc.construct (ctx, x, cmp);
    nextWL->get ().push_back (c);
  }

  template <typename F>
  void runCatching (F& func, Ctxt* c, UserCtxt& uhand) {
    try {
      func (c->getElem (), uhand);

    } catch (ConflictFlag f) {

      switch (f) {
        case CONFLICT: 
          c->disableSrc ();
          break;
        default:
          GALOIS_DIE ("can't handle conflict flag type");
          break;
      }
    }
  }


  template <typename S>
  class SafetyTestLoop {


    struct GetActive {
      const T& operator () (const Ctxt* c) const {
        assert (c != nullptr);
        return c->getElem ();
      }
    };

    S safetyTest;
    PerThreadVector<Ctxt*> safeSources;

  public:

    explicit UnstableSrcTestLoop (const S& safetyTest): safetyTest (safetyTest) {}

    template <typename I>
    std::pair <I, I> run (I beg, I end) {

      safeSources.clear_all ();
      Galois::do_all (beg, end,
          [this] (const Ctxt* c) {
            auto bt = boost::make_transform_iterator (beg, GetActive ());
            auto et = boost::make_transform_iterator (end, GetActive ());
            if (safetyTest (c->active, bt, et)) {
              safeSources.get ().push_back (c);
            }
          },
          "safety_test_loop");

      return std::make_pair (safeSources.begin_all (), safeSources.end_all ());
    }
  };

  template <>
  struct SafetyTestLoop<int> {

    template <typename I>
    std::pair<I, I> run (I beg, I end) const { 
      return std::make_pair (beg, end);
    }
  };



};

template <typename Iter, typename Cmp, typename NhFunc, typename OpFunc, typename ST>
void for_each_ordered_2p_param (Iter beg, Iter end, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const ST& safetyTest, const char* loopname=0) {

  using T = typename std::iterator_traits<Iter>::value_type;
  using Exec = TwoPhaseContext<T, Cmp, NhFunc, OpFunc, ST>;
  
  Exec e (cmp, nhFunc, opFunc, safetyTest);

  e.fill_initial (beg, end);
  e.execute ();
}

template <typename Iter, typename Cmp, typename NhFunc, typename OpFunc>
void for_each_ordered_2p_param (Iter beg, Iter end, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const char* loopname=0) {
  for_each_ordered_2p_param (beg, end, cmp, nhFunc, opFunc, int (0), loopname);
}


} // end namespace Runtime
} // end namespace Galois


#endif //  GALOIS_RUNTIME_KDG_PARAMETER_H
