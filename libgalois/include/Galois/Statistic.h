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
#include "Galois/Runtime/StatCollector.h"
#include "Galois/Runtime/Sampling.h"
#include "Galois/Timer.h"

#include "boost/utility.hpp"

#include <deque>
#include <limits>

namespace Galois {

/**
 * Basic per-thread statistics counter.
 */
template<typename Ty>
class StatisticBase {
  std::string statname;
  std::string loopname;
  Substrate::PerThreadStorage<std::pair<bool, Ty> > vals;

public:
  StatisticBase(const std::string& _sn, std::string _ln = "(NULL)", Ty init = Ty())
    : statname(_sn), loopname(_ln), vals(false,init) { }
  
  ~StatisticBase() {
    report();
  }

  //! Adds stat to stat pool, usually deconsructor or StatManager calls this for you.
  void report() {
    for (unsigned x = 0; x < vals.size(); ++x)  {
      auto* ptr = vals.getRemote(x);
      if (ptr->first)
        Galois::Runtime::reportStat(loopname, statname, ptr->second, x);
    }
  }

  StatisticBase& operator+=(Ty v) {
    auto* ptr = vals.getLocal();
    ptr->first = true;
    ptr->second += v;
    return *this;
  }

  Ty getVal() const {
    return vals.getLocal()->second;
  }

};
using Statistic = StatisticBase<unsigned long>;

#if 0 // Deprecated 
/**
 * Controls lifetime of stats. Users usually instantiate in main to print out
 * statistics at program exit.
 */
class StatManager: private boost::noncopyable {
  std::deque<Statistic*> stats;
  std::string statOutputFile_name;
  bool stat_printed;

public:
  StatManager(){
   statOutputFile_name = "";
   stat_printed = false;
  }
  StatManager(std::string name){
   statOutputFile_name = name;
   stat_printed = false;
  }
  ~StatManager() {
    if(!stat_printed){
      for(auto* s : stats)
        s->report();
      Galois::Runtime::printStats(statOutputFile_name);
    }
  }
  //! Statistics that are not lexically scoped must be added explicitly
  void push(Statistic& s) {
    stats.push_back(&s);
  }

  void reportStat(){
    for(auto* s : stats)
      s->report();
    Galois::Runtime::printStats(statOutputFile_name);
    stat_printed = true;
  }
};
#endif

//! Provides statistic interface around timer
// TODO: switch to const char* const or 
// a specialized "small string" type that doesn't allocate memory
// and compare with std::string to see if faster or equal
class StatTimer : public TimeAccumulator {
  std::string name;
  std::string loopname;
  bool main;
  bool valid;

protected:
  void init(const std::string& n, const std::string& l, bool m, bool s) {
    name = n;
    loopname = l;
    main = m;
    valid = false;
    if (s)
      start();
  }

public:
  StatTimer(const std::string& n) { init(n, "(NULL)", false, false); }
  StatTimer(const std::string& n, start_now_t t) { init(n, "(NULL)", false, true); }

  StatTimer(const std::string& n, const std::string& l) { init(n, l, false, false); }
  StatTimer(const std::string& n, const std::string& l, start_now_t t) { init(n, l, false, true); }

  StatTimer() { init("Time", "(NULL)", true, false); }
  StatTimer(start_now_t t) { init("Time", "(NULL)", true, true); }

  ~StatTimer() {
    if (valid)
      stop();
    if (TimeAccumulator::get()) // only report non-zero stat
      Galois::Runtime::reportStat(loopname, name, get(), Substrate::ThreadPool::getTID());
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

template <bool Enable> 
class CondStatTimer: public StatTimer {
public:
  CondStatTimer(const char* name): StatTimer("Time", name) {}
};

template <> class CondStatTimer<false> {
public:

  CondStatTimer(const char* name) {}

  void start(void) const {}

  void stop(void) const {}
};

template <typename F>
void timeThis(F& f, const char* const name) {
  StatTimer t("Time", name);

  t.start();

  f();

  t.stop();
}


template <bool enabled>
class PerThreadTimer {

protected:

  const char* const loopname;
  const char* const category;

  Substrate::PerThreadStorage<ThreadTimer<enabled> > timers;


  void reportTimes(void) {

    int64_t minTime = std::numeric_limits<int64_t>::max();

    std::string timeCat = category + std::string("-per-thread-times(ns)");

    for (unsigned i = 0; i < timers.size(); ++i) {

      auto ns = timers.getRemote(i)->get_nsec();
      Galois::Runtime::reportStat(loopname, timeCat.c_str(), ns, i);

      minTime = std::min(minTime, ns);
    }

    std::string lagCat = category + std::string("-per-thread-lag(ns)");
    for (unsigned i = 0; i < timers.size(); ++i) {

      auto ns = timers.getRemote(i)->get_nsec();
      auto lag = ns - minTime;
      assert(lag > 0 && "negative time lag from min is impossible");

      Galois::Runtime::reportStat(loopname, lagCat.c_str(), lag, i);
    }
  }

public:

  explicit PerThreadTimer(const char* const _loopname, const char* const _category)
    :
      loopname(_loopname),
      category(_category)
  {}

  ~PerThreadTimer(void) {
    reportTimes();
  }

  void start(void) {
    timers.getLocal()->start();
  }

  void stop(void) {
    timers.getLocal()->stop();
  }
};

template<> class PerThreadTimer<false> {

public:
  explicit PerThreadTimer(const char* const _loopname, const char* const _category)
  {}

  void start(void) const {}

  void stop(void) const {}

};


}
#endif
