/** KDG two phase executor -*- C++ -*-
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
 * @section Description
 *
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_KDG_TWO_PHASE_H
#define GALOIS_RUNTIME_KDG_TWO_PHASE_H

#include "Galois/Accumulator.h"
#include "Galois/Atomic.h"
#include "Galois/BoundedVector.h"
#include "Galois/gdeque.h"
#include "Galois/PriorityQueue.h"
#include "Galois/Timer.h"
#include "Galois/DoAllWrap.h"

#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/DoAll.h"
#include "Galois/Runtime/ForEachTraits.h"
#include "Galois/Runtime/ParallelWork.h"
#include "Galois/Runtime/PerThreadWorkList.h"
#include "Galois/Runtime/Range.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/KDGtwoPhaseSupport.h"
#include "Galois/Runtime/WindowWorkList.h"
#include "Galois/Runtime/UserContextAccess.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/ll/ThreadRWlock.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"
#include "Galois/Runtime/mm/Mem.h"

#include <boost/iterator/transform_iterator.hpp>

#include <iostream>

static cll::opt<double> commitRatioArg("cratio", cll::desc("target commit ratio for two phase executor"), cll::init(0.80));

namespace Galois {
namespace Runtime {

// TODO: figure out when to call startIteration

template <typename T, typename Cmp, typename NhFunc, typename OpFunc, typename WindowWL> class KDGtwoPhaseStableExecutor {

public:
  using Ctxt = TwoPhaseContext<T, Cmp>;
  using CtxtAlloc = Galois::Runtime::MM::FSBGaloisAllocator<Ctxt>;
  using CtxtWL = Galois::Runtime::PerThreadVector<Ctxt*>;

  using UserCtxt = UserContextAccess<T>;
  using PerThreadUserCtxt = PerThreadStorage<UserCtxt>;

protected:
  struct MakeContext {
    const Cmp& cmp;
    CtxtAlloc& ctxtAlloc;

    MakeContext (const Cmp& cmp, CtxtAlloc& ctxtAlloc)
      : cmp (cmp), ctxtAlloc (ctxtAlloc)
    {}

    Ctxt* operator () (const T& x) {
      Ctxt* ctx = ctxtAlloc.allocate (1);
      assert (ctx != nullptr);
      ctxtAlloc.construct (ctx, x, cmp);

      return ctx;
    }
  };

  Cmp cmp;
  NhFunc nhFunc;
  OpFunc opFunc;
  WindowWL winWL;
  CtxtAlloc ctxtAlloc;
  MakeContext ctxtMaker;
  PerThreadUserCtxt userHandles;


  size_t windowSize = 0;
  GAccumulator<size_t> numCommitted;

  size_t rounds = 0;
  size_t prevCommits = 0;
  GAccumulator<size_t> total;


public:

  KDGtwoPhaseStableExecutor (
      const Cmp& cmp, 
      const NhFunc& nhFunc,
      const OpFunc& opFunc)
    :
      cmp (cmp),
      nhFunc (nhFunc),
      opFunc (opFunc),
      winWL (cmp),
      ctxtMaker (cmp, ctxtAlloc)
  {}

  ~KDGtwoPhaseStableExecutor () {
    printStats ();
  }

  template <typename I>
  void fill_initial (I beg, I end) {
    winWL.initfill (beg, end);
  }

  void execute () {
    execute_stable ();
  }

protected:

  GALOIS_ATTRIBUTE_PROF_NOINLINE void spillAll (CtxtWL& wl) {
    Galois::on_each (
        [this, &wl] (const unsigned tid, const unsigned numT) {
          while (!wl[tid].empty ()) {
            Ctxt* c = wl[tid].back ();
            wl[tid].pop_back ();

            winWL.push (c->getElem ());
            c->~Ctxt ();
            ctxtAlloc.deallocate (c, 1);
          }
        });

    assert (wl.empty_all ());
    assert (!winWL.empty ());

  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void refill (CtxtWL* wl, size_t currCommits, size_t prevWindowSize) {

    const size_t INIT_MAX_ROUNDS = 500;
    const size_t THREAD_MULT_FACTOR = 16;

    const double TARGET_COMMIT_RATIO = commitRatioArg;

    const size_t MIN_WIN_SIZE = OpFunc::CHUNK_SIZE * getActiveThreads ();
    // const size_t MIN_WIN_SIZE = 2000000; // OpFunc::CHUNK_SIZE * getActiveThreads ();

    const size_t WIN_OVER_SIZE_FACTOR = 8;

    if (prevWindowSize == 0) {
      assert (currCommits == 0);

      // initial settings
      if (ForEachTraits<OpFunc>::NeedsPush) {
        windowSize = std::max (
            (winWL.initSize ()),
            (THREAD_MULT_FACTOR * MIN_WIN_SIZE));

      } else {
        windowSize = std::min (
            (winWL.initSize () / INIT_MAX_ROUNDS),
            (THREAD_MULT_FACTOR * MIN_WIN_SIZE));
      }


    } else {

      assert (windowSize > 0);

      double commitRatio = double (currCommits) / double (prevWindowSize);

      if (commitRatio >= TARGET_COMMIT_RATIO) {
        windowSize *= 2;
        // windowSize = int (windowSize * commitRatio/TARGET_COMMIT_RATIO); 
        // windowSize = windowSize + windowSize / 2;

      } else {

          windowSize = int (windowSize * commitRatio/TARGET_COMMIT_RATIO); 

        // if (commitRatio / TARGET_COMMIT_RATIO < 0.90) {
          // windowSize = windowSize - (windowSize / 10);
// 
        // } else {
          // windowSize = int (windowSize * commitRatio/TARGET_COMMIT_RATIO); 
        // }
      }

    }

    if (windowSize < MIN_WIN_SIZE) { 
      windowSize = MIN_WIN_SIZE;
    }

    assert (windowSize > 0);


    if (ForEachTraits<OpFunc>::NeedsPush) {

      if (winWL.empty () && (wl->size_all () > windowSize)) {
        // a case where winWL is empty and all the new elements were going into 
        // nextWL. When nextWL becomes bigger than windowSize, we must do something
        // to control efficiency. One solution is to spill all elements into winWL
        // and refill
        //

        spillAll (*wl);

      } else if (wl->size_all () > (WIN_OVER_SIZE_FACTOR * windowSize)) {
        // too many adds. spill to control efficiency
        spillAll (*wl);
      }
    }


    winWL.poll (*wl, windowSize, wl->size_all (), ctxtMaker);

    // std::cout << "Calculated Window size: " << windowSize << ", Actual: " << wl->size_all () << std::endl;

  }




  // TODO: use setjmp here
  template <typename F>
  static void runCatching (F& func, Ctxt* c, UserCtxt& uhand) {
    Galois::Runtime::setThreadContext (c);

    int result = 0;

#ifdef GALOIS_USE_LONGJMP
    if ((result = setjmp(hackjmp)) == 0) {
#else
    try {
#endif
      func (c->getElem (), uhand);

#ifdef GALOIS_USE_LONGJMP
    } else {
      // TODO
    }
#else 
    } catch (ConflictFlag f) {
      result = f;
    }
#endif

    switch (result) {
      case 0:
        break;
      case CONFLICT: 
        c->disableSrc ();
        break;
      default:
        GALOIS_DIE ("can't handle conflict flag type");
        break;
    }
    

    Galois::Runtime::setThreadContext (NULL);
  }

  
  GALOIS_ATTRIBUTE_PROF_NOINLINE void prepareRound (CtxtWL*& currWL, CtxtWL*& nextWL) {
    ++rounds;
    std::swap (currWL, nextWL);
    size_t prevWindowSize = nextWL->size_all ();
    nextWL->clear_all ();

    size_t currCommits = numCommitted.reduce () - prevCommits;
    prevCommits += currCommits;

    refill (currWL, currCommits, prevWindowSize);
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void expandNhood (CtxtWL* currWL) {

    Galois::do_all_choice (currWL->begin_all (), currWL->end_all (),
        [this] (Ctxt* c) {
          UserCtxt& uhand = *userHandles.getLocal ();
          uhand.reset ();

          // nhFunc (c, uhand);
          runCatching (nhFunc, c, uhand);

          total += 1;
        },
        "expandNhood");

  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void applyOperator (CtxtWL* currWL, CtxtWL* nextWL) {
    const T* minElem = nullptr;

    if (ForEachTraits<OpFunc>::NeedsPush) {
      if (!winWL.empty ()) {
        // make a copy
        minElem = new T (*winWL.getMin ());
      }
    }


    Galois::do_all_choice (currWL->begin_all (), currWL->end_all (),
        [this, minElem, nextWL] (Ctxt* c) {

          bool commit = false;

          UserCtxt& uhand = *userHandles.getLocal ();
          uhand.reset ();

          if (c->isSrc ()) {
          // opFunc (c->active, uhand);
          runCatching (opFunc, c, uhand);

          commit = c->isSrc (); // in case opFunc signalled abort

          } else {
          commit = false;
          }

          if (commit) {

          numCommitted += 1;

          if (ForEachTraits<OpFunc>::NeedsPush) { 


            for (auto i = uhand.getPushBuffer ().begin ()
                , endi = uhand.getPushBuffer ().end (); i != endi; ++i) {

              if (minElem == nullptr || !cmp (*minElem, *i)) {
                // if *i >= *minElem
                nextWL->get ().push_back (ctxtMaker (*i));

              } else {
                assert (minElem != nullptr);

                winWL.push (*i);

              } 
            }

          } else {
            assert (uhand.getPushBuffer ().begin () == uhand.getPushBuffer ().end ());
          }

          c->commitIteration ();

          c->~Ctxt ();
          ctxtAlloc.deallocate (c, 1);

          } else {
            c->cancelIteration ();
            c->reset ();
            nextWL->get ().push_back (c);

          }

        },
        "applyOperator");


    if (ForEachTraits<OpFunc>::NeedsPush) {
      if (minElem != nullptr) {
        delete minElem;
        minElem = nullptr;
      }
    }
  }


  void execute_stable () {

    CtxtWL* currWL = new CtxtWL ();
    CtxtWL* nextWL = new CtxtWL ();

    while (true) {

      prepareRound (currWL, nextWL);

      if (currWL->empty_all ()) {
        break;
      }

      expandNhood (currWL);

      applyOperator (currWL, nextWL);
      
    }

    delete currWL; currWL = nullptr;
    delete nextWL; nextWL = nullptr;

  }

  void printStats (void) {
    std::cout << "Two Phase Window executor, rounds: " << rounds << std::endl;
    std::cout << "Two Phase Window executor, commits: " << numCommitted.reduce () << std::endl;
    std::cout << "Two Phase Window executor, total: " << total.reduce () << std::endl;
    std::cout << "Two Phase Window executor, efficiency: " << double (numCommitted.reduce ()) / total.reduce () << std::endl;
    std::cout << "Two Phase Window executor, avg. parallelism: " << double (numCommitted.reduce ()) / rounds << std::endl;
  }
  
};


namespace impl {
  template <bool B, typename T1, typename T2> 
  struct ChooseIf {
    using Ret_ty = T1;
  };

  template <typename T1, typename T2>
  struct ChooseIf<false, T1, T2> {
    using Ret_ty = T2;
  };
} // end impl

template <typename Iter, typename Cmp, typename NhFunc, typename OpFunc>
void for_each_ordered_2p_win (Iter beg, Iter end, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const char* loopname=0) {

  using T = typename std::iterator_traits<Iter>::value_type;
  // using WindowWL = SortedRangeWindowWL<T, Cmp>;
  
  const bool ADDS = ForEachTraits<OpFunc>::NeedsPush;

  using WindowWL = typename impl::ChooseIf<ADDS, PQbasedWindowWL<T, Cmp>, SortedRangeWindowWL<T, Cmp> >::Ret_ty;

  using Exec = KDGtwoPhaseStableExecutor<T, Cmp, NhFunc, OpFunc, WindowWL>;
  
  Exec e (cmp, nhFunc, opFunc);

  e.fill_initial (beg, end);
  e.execute ();
}


template <typename T, typename Cmp, typename NhFunc, typename OpFunc, typename SL, typename WindowWL>

class KDGtwoPhaseUnstableExecutor: public KDGtwoPhaseStableExecutor<T, Cmp, NhFunc, OpFunc, WindowWL>  {

  using Base = KDGtwoPhaseStableExecutor<T, Cmp, NhFunc, OpFunc, WindowWL>;

  using Ctxt = typename Base::Ctxt;
  using CtxtWL = typename Base::CtxtWL;

  SL serialLoop;

public:
  KDGtwoPhaseUnstableExecutor (
      const Cmp& cmp, 
      const NhFunc& nhFunc,
      const OpFunc& opFunc,
      const SL& serialLoop)
    :
      Base (cmp, nhFunc, opFunc),
      serialLoop (serialLoop)
  {}


  void execute (void) {
    execute_unstable ();
  }

protected:

  struct GetActive: public std::unary_function<Ctxt*, const T&> {
    const T& operator () (const Ctxt* c) const {
      assert (c != nullptr);
      return c->getElem ();
    }
  };

  GALOIS_ATTRIBUTE_PROF_NOINLINE void expandNhood (CtxtWL* currWL) {

    auto m_beg = boost::make_transform_iterator (currWL->begin_all (), GetActive ());
    auto m_end = boost::make_transform_iterator (currWL->end_all (), GetActive ());

    using UserCtxt = typename Base::UserCtxt;

    NhFunc& func = Base::nhFunc;
    typename Base::PerThreadUserCtxt& uh = Base::userHandles;
    GAccumulator<size_t>& total = Base::total;

    Galois::do_all_choice (currWL->begin_all (), currWL->end_all (),
        [m_beg, m_end, &func, &uh, &total] (Ctxt* c) {
          UserCtxt& uhand = *uh.getLocal ();
          uhand.reset ();

          // nhFunc (c, uhand);
          // runCatching (nhFunc, c, uhand);
          Galois::Runtime::setThreadContext (c);
          int result = 0;
#ifdef GALOIS_USE_LONGJMP
          if ((result = setjmp(hackjmp)) == 0) {
#else
          try {
#endif 
            func (c->getElem (), uhand, m_beg, m_end);

#ifdef GALOIS_USE_LONGJMP
          } else {
            // nothing to do here
          }
#else
          } catch (ConflictFlag f) {
            result = f;
          }
#endif

          switch (result) {
            case 0:
              break;
            case CONFLICT: 
              c->disableSrc ();
              break;
            default:
              GALOIS_DIE ("can't handle conflict flag type");
              break;
          }
          

          Galois::Runtime::setThreadContext (NULL);

          total += 1;
        },
        "expandNhood");

  }

  void execute_unstable (void) {

    CtxtWL* currWL = new CtxtWL ();
    CtxtWL* nextWL = new CtxtWL ();

    while (true) {

      Base::prepareRound (currWL, nextWL);

      if (currWL->empty_all ()) {
        break;
      }

      expandNhood (currWL);

      for (auto i = currWL->begin_all ()
          , endi = currWL->end_all (); i != endi; ++i) {

        if ((*i)->isSrc ()) {
          serialLoop ((*i)->getElem ());
        }
      }

      Base::applyOperator (currWL, nextWL);
      
    }

    delete currWL; currWL = nullptr;
    delete nextWL; nextWL = nullptr;

  }

};

template <typename Iter, typename Cmp, typename NhFunc, typename OpFunc, typename SL>
void for_each_ordered_2p_win (Iter beg, Iter end, const Cmp& cmp, const NhFunc& nhFunc, const OpFunc& opFunc, const SL& serialLoop, const char* loopname=0) {

  using T = typename std::iterator_traits<Iter>::value_type;
  using WindowWL = PQbasedWindowWL<T, Cmp>;

  using Exec = KDGtwoPhaseUnstableExecutor<T, Cmp, NhFunc, OpFunc, SL, WindowWL>;
  
  Exec e (cmp, nhFunc, opFunc, serialLoop);

  e.fill_initial (beg, end);
  e.execute ();
}



} // end namespace Runtime
} // end namespace Galois

#endif //  GALOIS_RUNTIME_KDG_TWO_PHASE_H
