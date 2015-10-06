/** Statistic type -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
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

#ifndef GALOIS_STATISTIC_H
#define GALOIS_STATISTIC_H

#include "Galois/Substrate/PerThreadStorage.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/Sampling.h"
#include "Galois/Timer.h"

#include "boost/utility.hpp"

#include <deque>

namespace Galois {

/**
 * Basic per-thread statistics counter.
 */
class Statistic {
  std::string statname;
  std::string loopname;
  Substrate::PerThreadStorage<unsigned long> val;
  bool valid;

public:
  Statistic(const std::string& _sn, std::string _ln = "(NULL)"): statname(_sn), loopname(_ln), valid(true) { }

  ~Statistic() {
    report();
  }

  //! Adds stat to stat pool, usually deconsructor or StatManager calls this for you.
  void report() {
    if (valid)
      Galois::Runtime::reportStat(this);
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

/**
 * Controls lifetime of stats. Users usually instantiate in main to print out
 * statistics at program exit.
 */
class StatManager: private boost::noncopyable {
  std::deque<Statistic*> stats;

public:
  ~StatManager() {
    for (std::deque<Statistic*>::iterator ii = stats.begin(), ei = stats.end(); ii != ei; ++ii) {
      (*ii)->report();
    }
    Galois::Runtime::printStats();
  }

  //! Statistics that are not lexically scoped must be added explicitly
  void push(Statistic& s) {
    stats.push_back(&s);
  }
};

//! Flag type for {@link StatTimer}
struct start_now_t {};
constexpr start_now_t start_now = start_now_t();

//! Provides statistic interface around timer
class StatTimer : public TimeAccumulator {
  const char* name;
  const char* loopname;
  bool main;
  bool valid;

protected:
  void init(const char* n, const char* l, bool m, bool s) {
    name = n;
    loopname = l;
    main = m;
    valid = false;
    if (s)
      start();
  }

public:
  StatTimer(const char* n) { init(n, 0, false, false); }
  StatTimer(const char* n, start_now_t t) { init(n, 0, false, true); }

  StatTimer(const char* n, const char* l) { init(n, l, false, false); }
  StatTimer(const char* n, const char* l, start_now_t t) { init(n, l, false, true); }

  StatTimer() { init("Time", 0, true, false); }
  StatTimer(start_now_t t) { init("Time", 0, true, true); }

  ~StatTimer() {
    if (valid)
      stop();
    if (TimeAccumulator::get()) // only report non-zero stat
      Galois::Runtime::reportStat(loopname, name, get());
  }

  void start() {
    if (main)
      Galois::Runtime::beginSampling();
    TimeAccumulator::start();
    valid = true;
  }

  void stop() {
    valid = false;
    TimeAccumulator::stop();
    if (main)
      Galois::Runtime::endSampling();
  }
};

}
#endif
