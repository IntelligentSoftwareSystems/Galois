// Debug Worklists Wrappers -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

#ifndef __DEBUGWORKLIST_H_
#define __DEBUGWORKLIST_H_

#include <fstream>
#include <map>
#include "Galois/util/OnlineStats.h"

namespace GaloisRuntime {
namespace WorkList {

template<typename Indexer, typename realWL, typename T = int>
class WorkListTracker {
  struct p {
    OnlineStat stat;
    unsigned int epoch;
    std::map<unsigned int, OnlineStat> values;
  };

  //online collection of stats
  PerCPU<p> tracking;
  //global clock
  LL::CacheLineStorage<unsigned int> clock;
  //master thread counting towards a tick
  LL::CacheLineStorage<unsigned int> thread_clock;

  realWL wl;
  Indexer I;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef WorkListTracker<Indexer, typename realWL::template rethread<newconcurrent>::WL, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef WorkListTracker<Indexer, typename realWL::template retype<Tnew>::WL, Tnew> WL;
  };

  typedef T value_type;

  WorkListTracker()
  {
    clock.data = 0;
    thread_clock.data = 0;
  }

  ~WorkListTracker() {

    //First flush the stats
    for (unsigned int t = 0; t < tracking.size(); ++t) {
      p& P = tracking.get(t);
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
	p& P = tracking.get(t);
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
  bool push(value_type val) {
    return wl.push(val);
  }

  bool pushi(value_type val) {
    return wl.pushi(val);
  }

  std::pair<bool, value_type> pop() {
    std::pair<bool, value_type> ret = wl.pop();
    if (!ret.first) return ret;
    p& P = tracking.get();
    unsigned int cclock = clock.data;
    if (P.epoch != cclock) {
      if (P.stat.getCount())
	P.values[P.epoch] = P.stat;
      P.stat.reset();
      P.epoch = clock.data;
    }
    unsigned int index = I(ret.second);
    P.stat.insert(index);
    if (tracking.myEffectiveID() == 0) {
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

template<typename iWL>
class NoInlineFilter {
  iWL wl;

public:
  typedef typename iWL::value_type value_type;
  
  template<bool concurrent>
  struct rethread {
    typedef NoInlineFilter<typename iWL::template rethread<concurrent>::WL> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef NoInlineFilter<typename iWL::template retype<Tnew>::WL> WL;
  };

  //! push a value onto the queue
  bool push(value_type val) __attribute__((noinline)) {
    return wl.push(val);
  }

  bool pushi(value_type val) __attribute__((noinline)) {
    return wl.pushi(val);
  }

  std::pair<bool, value_type> pop() __attribute__((noinline)) {
    return wl.pop();
  }
};


}
}

#endif
