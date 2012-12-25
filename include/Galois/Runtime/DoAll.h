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

#include "Galois/Runtime/Barrier.h"

namespace GaloisRuntime {

struct EmptyFn {
  template<typename T>
  void operator()(T a, T b) {}
};

// TODO(ddn): Tune stealing. DMR suffers when stealing is on
// TODO: add loopname + stats
template<class FunctionTy, class ReduceFunTy, class RangeTy, bool useStealing=false>
class DoAllWork {
  typedef typename RangeTy::local_iterator local_iterator;
  LL::SimpleLock<true> reduceLock;
  FunctionTy origF;
  FunctionTy outputF;
  ReduceFunTy RF;
  bool needsReduce;
  RangeTy range;
  GBarrier barrier;

  struct SharedState {
    local_iterator stealBegin;
    local_iterator stealEnd;
    LL::SimpleLock<true> stealLock;
  };

  struct PrivateState {
    local_iterator begin;
    local_iterator end;
    FunctionTy F;
    PrivateState(FunctionTy& o) :F(o) {}
  };

  PerThreadStorage<SharedState> TLDS;

  //! Master execution function for this loop type
  void processRange(PrivateState& tld) {
    for (; tld.begin != tld.end; ++tld.begin)
      tld.F(*tld.begin);
  }

  bool doSteal(SharedState& source, PrivateState& dest) {
    //This may not be safe for iterators with complex state
    if (source.stealBegin != source.stealEnd) {
      source.stealLock.lock();
      if (source.stealBegin != source.stealEnd) {
	dest.begin = source.stealBegin;
	source.stealBegin = dest.end = Galois::split_range(source.stealBegin, source.stealEnd);
      }
      source.stealLock.unlock();
    }
    return dest.begin != dest.end;
  }

  void populateSteal(PrivateState& tld, SharedState& tsd) {
    if (tld.begin != tld.end && std::distance(tld.begin, tld.end) > 1) {
      tsd.stealLock.lock();
      tsd.stealEnd = tld.end;
      tsd.stealBegin = tld.end = Galois::split_range(tld.begin, tld.end);
      tsd.stealLock.unlock();
    }
  }

  GALOIS_ATTRIBUTE_NOINLINE
  bool trySteal(PrivateState& mytld) {
    //First try stealing from self
    if (doSteal(*TLDS.getLocal(), mytld))
      return true;
    //Then try stealing from neighbors
    unsigned myID = LL::getTID();
    for (unsigned x = 1; x < galoisActiveThreads; x += x) {
      SharedState& r = *TLDS.getRemote((myID + x) % galoisActiveThreads);
      if (doSteal(r, mytld)) {
	//populateSteal(mytld);
	return true;
      }
    }
    return false;
  }

  void doReduce(PrivateState& mytld) {
    if (needsReduce) {
      reduceLock.lock();
      RF(outputF, mytld.F);
      reduceLock.unlock();
    }
  }

public:
  DoAllWork(const FunctionTy& F, const ReduceFunTy& R, bool needsReduce, RangeTy r)
    : origF(F), outputF(F), RF(R), needsReduce(needsReduce), range(r)
  {
    barrier.reinit(galoisActiveThreads);
  }

  void operator()() {
    //Assume the copy constructor on the functor is readonly
    PrivateState thisTLD(origF);
    thisTLD.begin = range.local_begin();
    thisTLD.end = range.local_end();

    if (useStealing) {
      populateSteal(thisTLD, *TLDS.getLocal());

      // threads could start stealing from other threads whose
      // range has not been initialized yet
      barrier.wait();
    }

    do {
      processRange(thisTLD);
    } while (useStealing && trySteal(thisTLD));

    doReduce(thisTLD);
  }

  FunctionTy getFn() const { return outputF; }
};

template<typename RangeTy, typename FunctionTy, typename ReducerTy>
FunctionTy do_all_impl_dispatch(RangeTy range, FunctionTy f, ReducerTy r, bool needsReduce, std::random_access_iterator_tag) {
  // Still have no work stealing because some do_all loops are actually placing data.
  // TODO: differentiate calls or alternatively enrich Range objects to do the right thing
  DoAllWork<FunctionTy, ReducerTy, RangeTy, false> W(f, r, needsReduce, range);

  RunCommand w[2] = {Config::ref(W),
		     Config::ref(getSystemBarrier())};
  getSystemThreadPool().run(&w[0], &w[2]);
  return W.getFn();
}

template<typename RangeTy, typename FunctionTy, typename ReducerTy>
FunctionTy do_all_impl_dispatch(RangeTy range, FunctionTy f, ReducerTy r, bool needsReduce, std::input_iterator_tag) {
  DoAllWork<FunctionTy, ReducerTy, RangeTy, false> W(f, r, needsReduce, range);

  RunCommand w[2] = {Config::ref(W),
		     Config::ref(getSystemBarrier())};
  getSystemThreadPool().run(&w[0], &w[2]);
  return W.getFn();
}

template<typename RangeTy, typename FunctionTy, typename ReducerTy>
FunctionTy do_all_impl(RangeTy range, FunctionTy f, ReducerTy r, bool needsReduce) {
  if (GaloisRuntime::inGaloisForEach) {
    return std::for_each(range.begin(), range.end(), f);
  } else {
    inGaloisForEach = true;

    typename std::iterator_traits<typename RangeTy::local_iterator>::iterator_category category;
    FunctionTy retval(do_all_impl_dispatch(range, f, r, needsReduce, category));

    inGaloisForEach = false;

    return retval;
  }
}

//! Backdoor function to enable stealing in do_all
template<bool Steal, typename IterTy, typename FunctionTy>
void do_all_impl(IterTy b, IterTy e, FunctionTy f, const char* loopname=0) {
  if (GaloisRuntime::inGaloisForEach) {
    std::for_each(b, e, f);
  } else {
    inGaloisForEach = true;

    typedef StandardRange<IterTy> Range;
    DoAllWork<FunctionTy, EmptyFn, Range, Steal> W(f, EmptyFn(), false, makeStandardRange(b, e));
    RunCommand w[2] = {Config::ref(W),
                       Config::ref(getSystemBarrier())};
    getSystemThreadPool().run(&w[0], &w[2]);
    inGaloisForEach = false;
  }
}

} //namespace GaloisRuntime

#endif // GALOIS_RUNTIME_DOALL_H
