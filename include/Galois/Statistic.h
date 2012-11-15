/** Statistic type -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
#ifndef GALOIS_STATISTIC_H
#define GALOIS_STATISTIC_H

#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Sampling.h"
#include "Galois/Timer.h"

#include "boost/utility.hpp"

#include <list>

namespace Galois {

class Statistic {
  std::string statname;
  std::string loopname;
  GaloisRuntime::PerThreadStorage<unsigned long> val;
  bool valid;

public:
  Statistic(const std::string& _sn, unsigned long v, const std::string& _ln = "(NULL)"): statname(_sn), loopname(_ln), valid(true) {
    *val.getLocal() = v;
  }
  
  Statistic(const std::string& _sn, const std::string& _ln = "(NULL)"): statname(_sn), loopname(_ln), valid(true) { }

  ~Statistic() {
    report();
  }

  //! Adds stat to stat pool, usually deconsructor or StatManager calls this for you.
  void report() {
    if (valid)
      GaloisRuntime::reportStat(this);
    valid = false;
  }

  unsigned long getValue(unsigned tid) {
    return *val.getRemote(tid);
  }

  std::string& getLoopname() {
    return loopname;
  }

  std::string& getStatname() {
    return statname;
  }

  Statistic& operator+=(unsigned long v) {
    *val.getLocal() += v;
    return *this;
  }
};

//! Controls lifetime of stats. Users usually instantiate an instance in main.
class StatManager: private boost::noncopyable {
  std::list<Statistic*> stats;

public:
  ~StatManager() {
    for (std::list<Statistic*>::iterator ii = stats.begin(), ei = stats.end(); ii != ei; ++ii) {
      (*ii)->report();
    }
    GaloisRuntime::printStats();
  }

  //! Statistics that are not lexically scoped must be added explicitly
  void push(Statistic& s) {
    stats.push_back(&s);
  }
};

//! Provides statistic interface around timer
class StatTimer : public Timer {
  const char* name;
  const char* loopname;
  bool main;

public:
  StatTimer(): name("Time"), loopname(0), main(true) { }
  StatTimer(const char* n, const char* l = 0): name(n), loopname(l), main(false) { }
  ~StatTimer() {
    GaloisRuntime::reportStat(loopname, name, get());
    if (main)
      GaloisRuntime::reportSampling(loopname);
  }

  void start() {
    if (main)
      GaloisRuntime::beginSampling();
    Timer::start();
  }

  void stop() {
    Timer::stop();
    if (main)
      GaloisRuntime::endSampling();
  }
};

}

#endif
