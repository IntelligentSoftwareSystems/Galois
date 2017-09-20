/** Debug worklist wrappers-*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_WORKLISTDEBUG_H
#define GALOIS_RUNTIME_WORKLISTDEBUG_H

#include "galois/OnlineStats.h"

#include <fstream>
#include <map>

namespace galois {
namespace worklists {

template<typename Indexer, typename realWL, typename T = int>
class WorkListTracker {
  struct p {
    OnlineStat stat;
    unsigned int epoch;
    std::map<unsigned, OnlineStat> values;
  };

  //online collection of stats
  substrate::PerThreadStorage<p> tracking;
  //global clock
  substrate::CacheLineStorage<unsigned int> clock;
  //master thread counting towards a tick
  substrate::CacheLineStorage<unsigned int> thread_clock;

  realWL wl;
  Indexer I;

public:
  template<typename Tnew>
  using retype = WorkListTracker<Indexer, typename realWL::template retype<Tnew>, Tnew>;

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
    file << "Epoch,thread,type,value\n";
    //for each epoch
    for (unsigned int x = 0; x <= clock.data; ++x) {
      //for each thread
      for (unsigned int t = 0; t < tracking.size(); ++t) {
	p& P = *tracking.getRemote(t);
	if (P.values.find(x) != P.values.end()) {
	  OnlineStat& S = P.values[x];
          file << x << "," << t << ",count," << S.getCount() << "\n";
          file << x << "," << t << ",mean," << S.getMean() << "\n";
          file << x << "," << t << ",variance," << S.getVariance() << "\n";
          file << x << "," << t << ",stddev," << S.getStdDeviation() << "\n";
          file << x << "," << t << ",min," << S.getMin() << "\n";
          file << x << "," << t << ",max," << S.getMax() << "\n";
	}
      }
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

  galois::optional<value_type> pop() {
    galois::optional<value_type> ret = wl.pop();
    if (!ret) return ret;
    p& P = *tracking.getLocal();
    unsigned int cclock = clock.data;
    if (P.epoch != cclock) {
      if (P.stat.getCount())
	P.values[P.epoch] = P.stat;
      P.stat.reset();
      P.epoch = clock.data;
    }
    auto index = I(*ret);
    P.stat.insert(index);
    if (substrate::ThreadPool::getTID() == 0) {
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
  substrate::PerThreadStorage<p> tracking;

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
    unsigned tid = substrate::ThreadPool::getTID();
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
  using rethread = LoadBalanceTracker<typename realWL::template rethread<newconcurrent>, perEpoch>;

  template<typename Tnew>
  using retype = LoadBalanceTracker<typename realWL::template retype<Tnew>, perEpoch>;

  typedef typename realWL::value_type value_type;

  LoadBalanceTracker() :Pr(galois::getActiveThreads()) {}

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

  galois::optional<value_type> pop() {
    p& P = *tracking.getLocal();

    if (P.epoch != P.newEpoch)
      updateEpoch(P);

    galois::optional<value_type> ret = wl.pop();
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
  using rethread = NoInlineFilter<typename iWL::template rethread<concurrent> >;

  template<typename Tnew>
  using retype = NoInlineFilter<typename iWL::template retype<Tnew> >;

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
  galois::optional<value_type> pop()  {
    return wl.pop();
  }
};


}
} // end namespace galois

#endif
