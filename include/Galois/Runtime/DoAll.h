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

namespace GaloisRuntime {

struct dummyFN {
  template<typename T>
  void operator()(T a, T b) {}
};

// TODO(ddn): Tune stealing. DMR suffers when stealing is on
template<class FunctionTy, class ReduceFunTy, class IterTy, bool useStealing=false>
class DoAllWork {
  LL::SimpleLock<true> reduceLock;
  FunctionTy origF;
  ReduceFunTy RF;
  bool needsReduce;
  IterTy masterBegin;
  IterTy masterEnd;

  struct SharedState {
    IterTy stealBegin;
    IterTy stealEnd;
    LL::SimpleLock<true> stealLock;
  };

  struct PrivateState {
    IterTy begin;
    IterTy end;
    FunctionTy F;
    PrivateState(FunctionTy& o) :F(o) {}
  };

  PerThreadStorage<SharedState> TLDS;

  //! Master execution function for this loop type
  void processRange(PrivateState& tld) {
    for ( ; tld.begin != tld.end; ++tld.begin)
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
      RF(origF, mytld.F);
      reduceLock.unlock();
    }
  }

public:

  DoAllWork(const FunctionTy& F, const ReduceFunTy R, IterTy begin, IterTy end)
    : origF(F), RF(R), needsReduce(true), masterBegin(begin), masterEnd(end)
  {}

  DoAllWork(FunctionTy& F, IterTy begin, IterTy end)
    : origF(F), RF(dummyFN()), needsReduce(false), masterBegin(begin), masterEnd(end)
  {}

  void operator()() {
    //Assume the copy constructor on the functor is readonly
    PrivateState thisTLD(origF);
    std::pair<IterTy, IterTy> r = Galois::block_range(masterBegin, masterEnd, LL::getTID(), galoisActiveThreads);
    thisTLD.begin = r.first;
    thisTLD.end = r.second;

    if (useStealing)
      populateSteal(thisTLD, *TLDS.getLocal());

    do {
      processRange(thisTLD);
    } while (useStealing && trySteal(thisTLD));

    doReduce(thisTLD);
  }

  FunctionTy getFn() const { return origF; }

};

template<typename IterTy, typename FunctionTy>
FunctionTy do_all_impl(IterTy b, IterTy e, FunctionTy f) {
  assert(!inGaloisForEach);

  inGaloisForEach = true;

  DoAllWork<FunctionTy, dummyFN, IterTy> W(f, b, e);

  RunCommand w[2] = {Config::ref(W),
		     Config::ref(getSystemBarrier())};
  getSystemThreadPool().run(&w[0], &w[2]);

  inGaloisForEach = false;
  return W.getFn();
}

template<typename IterTy, typename FunctionTy, typename ReducerTy>
FunctionTy do_all_impl(IterTy b, IterTy e, FunctionTy f, ReducerTy r) {
  assert(!inGaloisForEach);

  inGaloisForEach = true;

  DoAllWork<FunctionTy, ReducerTy, IterTy> W(f, r, b, e);

  RunCommand w[2] = {Config::ref(W),
		     Config::ref(getSystemBarrier())};
  getSystemThreadPool().run(&w[0], &w[2]);

  inGaloisForEach = false;
  return W.getFn();
}

} //namespace GaloisRuntime

#endif // GALOIS_RUNTIME_DOALL_H
