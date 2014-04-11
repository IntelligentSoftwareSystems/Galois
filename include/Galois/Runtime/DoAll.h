/** Galois Simple Parallel Loop -*- C++ -*-
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
 * Implementation of the Galois foreach iterator. Includes various 
 * specializations to operators to reduce runtime overhead.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_DOALL_H
#define GALOIS_RUNTIME_DOALL_H

#include "Galois/gstl.h"
#include "Galois/Statistic.h"
#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Range.h"
#include "Galois/Runtime/ForEachTraits.h"

#include <algorithm>

namespace Galois {
namespace Runtime {

struct EmptyFn {
  template<typename T>
  void operator()(T a, T b) {}
};


template<bool doSteal, typename iterator>
struct doAllStealer;

template<typename iterator>
struct doAllStealer<false, iterator> {
  void populateSteal(iterator&, iterator&) {}
  bool trySteal(iterator&, iterator&) { return false; }
};

template<typename iterator>
struct doAllStealer<true, iterator> {

  struct state {
    iterator stealBegin;
    iterator stealEnd;
    LL::SimpleLock stealLock;
    state() { stealLock.lock(); }
  };

  PerThreadStorage<state> TLDS;

  // bool doSteal(SharedState& source, PrivateState& dest) {
  //   //This may not be safe for iterators with complex state
  //   if (source.stealBegin != source.stealEnd) {
  //     source.stealLock.lock();
  //     if (source.stealBegin != source.stealEnd) {
  //       dest.begin = source.stealBegin;
  //       source.stealBegin = dest.end = Galois::split_range(source.stealBegin, source.stealEnd);
  //     }
  //     source.stealLock.unlock();
  //   }
  //   return dest.begin != dest.end;
  // }

  void populateSteal(iterator& begin, iterator& end) {
    state& tld = *TLDS.getLocal();
    if (std::distance(begin, end) > 1) {
      tld.stealEnd = end;
      tld.stealBegin = end = Galois::split_range(begin, end);
      tld.stealLock.unlock();
    }
  }

  GALOIS_ATTRIBUTE_NOINLINE
  bool trySteal(iterator& begin, iterator& end) {
    // //First try stealing from self
    // if (doSteal(*TLDS.getLocal(), mytld))
    //   return true;
    // //Then try stealing from neighbors
    // unsigned myID = LL::getTID();
    // for (unsigned x = 1; x < activeThreads; x += x) {
    //   SharedState& r = *TLDS.getRemote((myID + x) % activeThreads);
    //   if (doSteal(r, mytld)) {
    //     //populateSteal(mytld);
    //     return true;
    //   }
    // }
    return false;
  }

};


// TODO(ddn): Tune stealing. DMR suffers when stealing is on
// TODO: add loopname + stats
template<class FunctionTy, class ReduceFunTy, class RangeTy, bool useStealing>
class DoAllWork {
  typedef typename RangeTy::local_iterator iterator;
  LL::SimpleLock reduceLock;
  FunctionTy origF;
  FunctionTy outputF;
  ReduceFunTy RF;
  RangeTy range;
  bool needsReduce;

  doAllStealer<useStealing, iterator> stealing;

public:
  DoAllWork(const FunctionTy& F, const ReduceFunTy& R, bool needsReduce, RangeTy r)
    : origF(F), outputF(F), RF(R), range(r), needsReduce(needsReduce)
  { }

  void operator()() {
    //Assume the copy constructor on the functor is readonly
    FunctionTy lFn(origF);
    iterator begin = range.local_begin();
    iterator end = range.local_end();

    stealing.populateSteal(begin,end);

    do {
      while (begin != end)
        lFn(*begin++);
    } while (stealing.trySteal(begin,end));

    if (needsReduce) {
      std::lock_guard<LL::SimpleLock> lg(reduceLock);
      RF(outputF, lFn);
    }
  }

  FunctionTy getFn() const { return outputF; }
};

template<typename RangeTy, typename FunctionTy, typename ReducerTy>
FunctionTy do_all_dispatch(RangeTy range, FunctionTy f, ReducerTy r, bool doReduce, const char* loopname, bool steal) {
  if (Galois::Runtime::inGaloisForEach) {
    return std::for_each(range.begin(), range.end(), f);
  } else {

    StatTimer LoopTimer("LoopTime", loopname);
    if (ForEachTraits<FunctionTy>::NeedsStats)
      LoopTimer.start();

    inGaloisForEach = true;
    FunctionTy retval;
    if (steal || doReduce) {
      DoAllWork<FunctionTy, ReducerTy, RangeTy, true> W(f, r, doReduce, range);
      getSystemThreadPool().run(activeThreads, std::ref(W));
      retval = W.getFn();
    } else {
      getSystemThreadPool().run(activeThreads, [&f, &range] () {
          auto begin = range.local_begin();
          auto end = range.local_end();
          while (begin != end)
            f(*begin++);
        });
      retval = f;
    }

    if (ForEachTraits<FunctionTy>::NeedsStats)  
      LoopTimer.stop();
    inGaloisForEach = false;
    return retval;
  }
}

template<typename RangeTy, typename FunctionTy>
FunctionTy do_all_impl(RangeTy range, FunctionTy f, const char* loopname = 0, bool steal = false) {
  return do_all_dispatch(range, f, EmptyFn(), false, loopname, steal);
}

template<typename RangeTy, typename FunctionTy, typename ReduceTy>
FunctionTy do_all_impl(RangeTy range, FunctionTy f, ReduceTy r, const char* loopname = 0, bool steal = false) {
  return do_all_dispatch(range, f, r, true, loopname, steal);
}

} // end namespace Runtime
} // end namespace Galois

#endif // GALOIS_RUNTIME_DOALL_H
