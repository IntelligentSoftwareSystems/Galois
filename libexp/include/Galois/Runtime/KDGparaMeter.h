/** KDG parameter based on two phase executor -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a gramework to exploit
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
#include "Galois/AltBag.h"
#include "Galois/DoAllWrap.h"
#include "Galois/PerThreadContainer.h"

#include "Galois/Substrate/Barrier.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Executor_DoAll.h"
#include "Galois/Runtime/Range.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Substrate/Termination.h"
#include "Galois/Substrate/ThreadPool.h"
#include "Galois/Runtime/KDGtwoPhaseSupport.h"
#include "Galois/Runtime/UserContextAccess.h"
#include "Galois/Substrate/gio.h"
#include "Galois/Runtime/ThreadRWlock.h"
#include "Galois/Substrate/CompilerSpecific.h"
#include "Galois/Runtime/Mem.h"

#include <boost/iterator/transform_iterator.hpp>

#include <iostream>

// TODO: figure out when to call startIteration

namespace Galois {
namespace Runtime {

template <typename T, typename Cmp, typename NhFunc, typename OpFunc, typename ST>
class KDGtwoPhaseParaMeter {

  using Ctxt = TwoPhaseContext<T, Cmp>;
  using CtxtAlloc = FixedSizeAllocator<Ctxt>;
  // using CtxtWL = PerThreadVector<Ctxt*>;
  using CtxtWL = PerThreadBag<Ctxt*>;

  using UserCtx = UserContextAccess<T>;
  using PerThreadUserCtx = Substrate::PerThreadStorage<UserCtx>;

  static const unsigned DEFAULT_CHUNK_SIZE = 16;

  Cmp cmp;
  NhFunc nhFunc;
  OpFunc opFunc;
  SafetyTestLoop<Ctxt, ST> stloop;
  CtxtAlloc ctxtAlloc;
  PerThreadUserCtx userHandles;

  CtxtWL* currWL = new CtxtWL ();

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
  }



  template <typename R>
  void fill_initial (const R& range) {
    Galois::Runtime::do_all_impl (range,
        [this] (const T& x) {
          createContext (x);
        },
        "fill_initial");
  }


  void execute () {


    size_t rounds = 0;
    GAccumulator<size_t> numCommitted;
    size_t totalIterations = 0;
    CtxtWL allSources;

    while (!currWL->empty_all ()) {

      ++rounds;

      totalIterations += currWL->size_all ();

      // expand nhood
      Galois::do_all_choice (
          makeLocalRange (*currWL),
          [this] (Ctxt* c) {
            UserCtx& uhand = *userHandles.getLocal ();
            uhand.reset ();

            // nhFunc (c, uhand);
            runCatching (nhFunc, c, uhand);
          },
          "expand_nhood",
          Galois::chunk_size<NhFunc::CHUNK_SIZE> ());



      // collect sources
      // release locks here to save a separate phase
      allSources.clear_all_parallel ();
      Galois::do_all_choice (
          makeLocalRange (*currWL),
          [this, &allSources] (Ctxt* c) {
            if (c->isSrc ()) {
              allSources.get ().push_back (c);

            } else {
              c->cancelIteration ();
            }
          },
          "collect_sources",
          Galois::chunk_size<DEFAULT_CHUNK_SIZE> ());

      // NOTE: safety test should be applied to all sources
      // where each source is tested against all elements earlier in priority
      // apply stability test
      stloop.run (makeLocalRange (allSources));

      // apply operator 
      Galois::do_all_choice (
          makeLocalRange (allSources),
          [this, &numCommitted] (Ctxt* c) {
            // assert (c->isSrc ());

            UserCtx& uhand = *userHandles.getLocal ();
            if (c->isSrc ()) {
              // opFunc (c->active, uhand);
              runCatching (opFunc, c, uhand);
            }

            // TODO: double check logic here
            if (c->isSrc ()) {

              numCommitted += 1;

              for (auto i = uhand.getPushBuffer ().begin ()
                , endi = uhand.getPushBuffer ().end (); i != endi; ++i) {
                
                createContext (*i);
              }

              // std::printf ("commit iteration %p\n", c);
              c->commitIteration ();

            } else {
              c->cancelIteration ();
            }

            uhand.reset ();
          },
          "apply_operator",
          Galois::chunk_size<OpFunc::CHUNK_SIZE> ());

      Galois::Runtime::on_each_impl (
          [this] (const unsigned tid, const unsigned numT) {
            typename CtxtWL::Cont_ty& cont  = currWL->get(tid); 

            for (auto i = cont.begin (), i_end = cont.end (); i != i_end;) { 

              if ((*i)->isSrc ()) {
                assert (!cont.empty ());
                // std::printf ("going to destroy and deallocate %p, list size=%zd, back=%p, i=%d\n", 
                  // *i, cont.size (), cont.back (), std::distance (cont.begin (), i));
// 
                auto last = cont.end ();
                --last;

                std::swap (*i, *last);

                Ctxt* src = *last;
                assert (src->isSrc ());
                cont.pop_back ();


                ctxtAlloc.destroy (src);
                ctxtAlloc.deallocate (src, 1);

                if (i == last) {
                  break;
                }

                i_end = cont.end ();

              } else {
                (*i)->reset ();
                ++i;
              }
            }

          },
          "clean-up");

    } // end while

    std::cout << "KDGParaMeter: number of rounds: " << rounds << std::endl;
    std::cout << "KDGParaMeter: total iterations: " << totalIterations << std::endl;
    std::cout << "KDGParaMeter: average parallelism: " << double (numCommitted.reduceRO ())/double(rounds) << std::endl;
    std::cout << "KDGParaMeter: parallelism as a fraction of total: " << double (numCommitted.reduceRO ())/double(totalIterations) << std::endl;

    allSources.clear_all_parallel ();
  }

private:

  void createContext (const T& x) {
    Ctxt* ctx = ctxtAlloc.allocate (1);
    assert (ctx != nullptr);
    ctxtAlloc.construct (ctx, x, cmp);
    currWL->get ().push_back (ctx);
  }

  // template <typename F>
  // void runCatching (F& func, Ctxt* c, UserCtx& uhand) {
// 
    // Galois::Runtime::setThreadContext (c);
// 
    // try {
      // func (c->getElem (), uhand);
// 
    // } catch (ConflictFlag f) {
// 
      // switch (f) {
        // case CONFLICT: 
          // c->disableSrc ();
          // break;
        // default:
          // GALOIS_DIE ("can't handle conflict flag type");
          // break;
      // }
    // }
  // }




};

template <typename R, typename Cmp, typename NhFunc, typename OpFunc, typename ST>
void for_each_ordered_2p_param (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const ST& safetyTest, const char* loopname) {

  using T = typename R::value_type;
  using Exec = KDGtwoPhaseParaMeter <T, Cmp, NhFunc, OpFunc, ST>;
  
  Exec e (cmp, nhFunc, opFunc, safetyTest);

  e.fill_initial (range);
  e.execute ();
}

template <typename R, typename Cmp, typename NhFunc, typename OpFunc>
void for_each_ordered_2p_param (const R& range, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const char* loopname) {
  for_each_ordered_2p_param (range, cmp, nhFunc, opFunc, int (0), loopname);
}


} // end namespace Runtime
} // end namespace Galois


#endif //  GALOIS_RUNTIME_KDG_PARAMETER_H
