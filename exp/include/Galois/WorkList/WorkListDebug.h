/** Debug worklist wrappers-*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_WORKLISTDEBUG_H
#define GALOIS_RUNTIME_WORKLISTDEBUG_H

#include "Galois/OnlineStats.h"

#include <fstream>
#include <map>

namespace Galois {
namespace WorkList {

template<typename Indexer, typename realWL, typename T >
class WorkListTracker {
  struct p {
    OnlineStat stat;
    unsigned int epoch;
    std::map<unsigned int, OnlineStat> values;
  };

  //online collection of stats
  Runtime::PerThreadStorage<p> tracking;
  //global clock
  Runtime::LL::CacheLineStorage<unsigned int> clock;
  //master thread counting towards a tick
  Runtime::LL::CacheLineStorage<unsigned int> thread_clock;

  realWL wl;
  Indexer I;

public:
  template<bool newconcurrent>
  struct rethread { typedef WorkListTracker<Indexer, typename realWL::template rethread<newconcurrent>::type, T> type; };

  template<typename Tnew>
  struct retype { typedef WorkListTracker<Indexer, typename realWL::template retype<Tnew>::type, Tnew> type; };

  typedef T value_type;

  WorkListTracker()
  {
    clock.data = 0;
    thread_clock.data = 0;
  }

  ~WorkListTracker() {

    //First flush the stats
    for (unsigned int t = 0; t < tracking.size(); ++t) {
      p& P = *tracking.getRemote(t);
      if (P.stat.getCount()) {
	P.values[P.epoch] = P.stat;
      }
    }

    std::ofstream file("tracking.csv", std::ofstream::app);

    //print header
    file << "Epoch";
    for (unsigned int t = 0; t < tracking.size(); ++t)
      file << "," << t << "_count,"
	   << t << "_mean,"
	   << t << "_variance,"
	   << t << "_stddev";
    file << "\n";

    //for each epoch
    for (unsigned int x = 0; x <= clock.data; ++x) {
      file << x;
      //for each thread
      for (unsigned int t = 0; t < tracking.size(); ++t) {
	p& P = *tracking.getRemote(t);
	if (P.values.find(x) != P.values.end()) {
	  OnlineStat& S = P.values[x];
	  file << "," << S.getCount()
	       << "," << S.getMean()
	       << "," << S.getVariance()
	       << "," << S.getStdDeviation();
	} else {
	  file << ",,,,";
	}
      }
      file << "\n";
    }
  }
    
  //! push a value onto the queue
  void push(value_type val) {
    wl.push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    wl.push(b,e);
  }

  template<typename RangeTy>
  void push_initial(const RangeTy& range) {
    wl.push_initial(range);
  }

  Galois::optional<value_type> pop() {
    Galois::optional<value_type> ret = wl.pop();
    if (!ret) return ret;
    p& P = *tracking.getRemote();
    unsigned int cclock = clock.data;
    if (P.epoch != cclock) {
      if (P.stat.getCount())
	P.values[P.epoch] = P.stat;
      P.stat.reset();
      P.epoch = clock.data;
    }
    unsigned int index = I(*ret);
    P.stat.insert(index);
    if (Runtime::LL::getTID() == 0) {
      ++thread_clock.data;
      if (thread_clock.data == 1024*10) {
	thread_clock.data = 0;
	clock.data += 1; //only on thread updates
	//__sync_fetch_and_add(&clock.data, 1);
      }
    }
    return ret;
  }
};

template<typename realWL, unsigned perEpoch = 1024>
class LoadBalanceTracker {
  struct p {
    unsigned int epoch;
    unsigned int newEpoch;
    std::vector<unsigned> values;
    p() :epoch(0), newEpoch(0), values(1) {}
  };

  //online collection of stats
  Runtime::PerThreadStorage<p> tracking;

  realWL wl;
  unsigned Pr;

  void atomic_max(unsigned* loc, unsigned val) {
    unsigned oe = *loc;
    while (oe < val) {
      oe = __sync_val_compare_and_swap(loc, oe, val);
    }
  }

  void updateEpoch(p& P) {
    unsigned multiple = 2;
    P.epoch = P.newEpoch;
    P.values.resize(P.epoch+1);
    unsigned tid = Runtime::LL::getTID();
    for (unsigned i = 1; i <= multiple; ++i) {
      unsigned n = tid * multiple + i;
      if (n < Pr)
	atomic_max(&tracking.getRemote(n)->newEpoch, P.epoch);
    }
  }

  void proposeEpoch(unsigned e) {
    atomic_max(&tracking.getRemote(0)->newEpoch, e);
  }

public:
  template<bool newconcurrent>
  struct rethread { typedef LoadBalanceTracker<typename realWL::template rethread<newconcurrent>::type, perEpoch> type; };

  template<typename Tnew>
  struct retype { typedef LoadBalanceTracker<typename realWL::template retype<Tnew>::type, perEpoch> type; };

  typedef typename realWL::value_type value_type;

  LoadBalanceTracker() :Pr(Galois::getActiveThreads()) {}

  ~LoadBalanceTracker() {
    std::ofstream file("tracking.csv", std::ofstream::trunc);

    //print header
    file << "Epoch";
    for (unsigned int t = 0; t < Pr; ++t)
      file << ",Thread " << t;
    file << "\n";

    unsigned maxEpoch = 0;
    for (unsigned int t = 0; t < Pr; ++t)
      maxEpoch = std::max(maxEpoch, (unsigned)tracking.getRemote(t)->values.size());

    //for each epoch
    for (unsigned int x = 0; x < maxEpoch; ++x) {
      file << x;
      //for each thread
      for (unsigned int t = 0; t < Pr; ++t)
	if (x < tracking.getRemote(t)->values.size())
	  file << "," << tracking.getRemote(t)->values[x];
	else
	  file << ",0";
      file << "\n";
    }
  }
    
  //! push a value onto the queue
  void push(value_type val) {
    wl.push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    wl.push(b,e);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    wl.push_initial(range);
  }

  Galois::optional<value_type> pop() {
    p& P = *tracking.getLocal();

    if (P.epoch != P.newEpoch)
      updateEpoch(P);

    Galois::optional<value_type> ret = wl.pop();
    if (!ret) return ret;
    unsigned num = ++P.values[P.epoch];
    if (num >= perEpoch)
      proposeEpoch(P.epoch + 1);
    return ret;
  }
};

template<typename iWL>
class NoInlineFilter {
  iWL wl;

public:
  typedef typename iWL::value_type value_type;
  
  template<bool concurrent>
  struct rethread { typedef NoInlineFilter<typename iWL::template rethread<concurrent>::type> type; };

  template<typename Tnew>
  struct retype { typedef NoInlineFilter<typename iWL::template retype<Tnew>::type > type; };

  //! push a value onto the queue
  GALOIS_ATTRIBUTE_NOINLINE
  void push(value_type val) {
    wl.push(val);
  }

  //These cannot have noinline in gcc, which makes this semi-useless
  template<typename Iter>
  GALOIS_ATTRIBUTE_NOINLINE
  void push(Iter b, Iter e) {
    wl.push(b,e);
  }

  //These cannot have noinline in gcc, which makes this semi-useless
  template<typename RangeTy>
  GALOIS_ATTRIBUTE_NOINLINE
  void push_initial(const RangeTy& range) {
    wl.push_initial(range);
  }

  GALOIS_ATTRIBUTE_NOINLINE
  Galois::optional<value_type> pop()  {
    return wl.pop();
  }
};


}
} // end namespace Galois

#endif
