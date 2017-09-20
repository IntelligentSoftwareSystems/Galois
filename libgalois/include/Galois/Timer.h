/** Simple timer support -*- C++ -*-
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

#ifndef GALOIS_TIMER_H
#define GALOIS_TIMER_H

#include "Galois/Runtime/Statistics.h"

#include "boost/utility.hpp"

#include <chrono>
#include <limits>

#include <cstdio>
#include <ctime>

namespace galois {

//! Flag type for {@link StatTimer}
struct start_now_t {};
constexpr start_now_t start_now = start_now_t();

//! A simple timer
class Timer: private boost::noncopyable {
  typedef std::chrono::steady_clock clockTy;
  //typedef std::chrono::high_resolution_clock clockTy;
  std::chrono::time_point<clockTy> startT, stopT;
public:
  Timer() = default;
  Timer(const start_now_t) { start(); }
  void start();
  void stop();
  unsigned long get() const;
  unsigned long get_usec() const;
};

//! A multi-start time accumulator.
//! Gives the final runtime for a series of intervals
class TimeAccumulator: private boost::noncopyable {
  Timer ltimer;
  unsigned long acc;
public:
  TimeAccumulator();
  void start();
  //!adds the current timed interval to the total
  void stop(); 
  unsigned long get() const;
  TimeAccumulator& operator+=(const TimeAccumulator& rhs);
  TimeAccumulator& operator+=(const Timer& rhs);
};

template <bool enabled> 
class ThreadTimer: private boost::noncopyable {
  timespec m_start;
  timespec m_stop;
  int64_t  m_nsec;

public:
  ThreadTimer (): m_nsec (0) {};

  void start (void) {
    clock_gettime (CLOCK_THREAD_CPUTIME_ID, &m_start);
  }

  void stop (void) {
    clock_gettime (CLOCK_THREAD_CPUTIME_ID, &m_stop);
    m_nsec += (m_stop.tv_nsec - m_start.tv_nsec);
    m_nsec += ((m_stop.tv_sec - m_start.tv_sec) << 30); // multiply by 1G
  }

  int64_t get_nsec(void) const { return m_nsec; }

  int64_t get_sec(void) const { return (m_nsec >> 30); }

  int64_t get_msec(void) const { return (m_nsec >> 20); }
    
};

template <>
class ThreadTimer<false> {
public:
  void start (void) const  {}
  void stop (void) const  {}
  int64_t get_nsec (void) const { return 0; }
  int64_t get_sec (void) const  { return 0; }
  int64_t get_msec (void) const  { return 0; }
};


//! Galois Timer that automatically reports stats upon destruction
//! Provides statistic interface around timer
class StatTimer : public TimeAccumulator {
  const char* name;
  const char* region;
  bool valid;

protected:
  void init(const char* const n, const char* const l, bool s) {
    name = n ? n : "Time";
    region = l? l : "(NULL)";
    valid = false;
    if (s)
      start();
  }

public:
  StatTimer(const char* const n) { init(n, nullptr, false); }
  StatTimer(const char* const n, start_now_t t) { init(n, nullptr, true); }

  StatTimer(const char* const n, const char* const l) { init(n, l, false); }
  StatTimer(const char* const n, const char* const l, start_now_t t) { init(n, l, true); }

  StatTimer() { init(nullptr, nullptr, false); }
  StatTimer(start_now_t t) { init(nullptr, nullptr, true); }

  ~StatTimer() {
    if (valid)
      stop();
    if (TimeAccumulator::get()) // only report non-zero stat
      galois::runtime::reportStat_Tmax(region, name, get());
  }

  void start() {
    TimeAccumulator::start();
    valid = true;
  }

  void stop() {
    valid = false;
    TimeAccumulator::stop();
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
class PerThreadTimer: private boost::noncopyable {

protected:

  const char* const region;
  const char* const category;

  substrate::PerThreadStorage<ThreadTimer<enabled> > timers;


  void reportTimes(void) {

    int64_t minTime = std::numeric_limits<int64_t>::max();


    for (unsigned i = 0; i < timers.size(); ++i) {

      auto ns = timers.getRemote(i)->get_nsec();

      minTime = std::min(minTime, ns);
    }

    std::string timeCat = category + std::string("-per-thread-times(ns)");
    std::string lagCat = category + std::string("-per-thread-lag(ns)");

    galois::substrate::getThreadPool(galois::getActiveThreads(),
        [&] (void) {
          auto ns = timers.getLocal()->get_nsec();
          auto lag = ns - minTime;
          assert(lag > 0 && "negative time lag from min is impossible");

          galois::runtime::reportStat_Tmax(region, timeCat.c_str(), ns);
          galois::runtime::reportStat_Tmax(region, lagCat.c_str(), lag);
        });

    // for (unsigned i = 0; i < timers.size(); ++i) {
// 
      // auto ns = timers.getRemote(i)->get_nsec();
      // auto lag = ns - minTime;
      // assert(lag > 0 && "negative time lag from min is impossible");
// 
      // galois::runtime::reportStat(region, lagCat.c_str(), lag, i);
      // galois::runtime::reportStat(region, timeCat.c_str(), ns, i);
    // }
  }

public:

  explicit PerThreadTimer(const char* const _region, const char* const _category)
    :
      region(_region),
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
  explicit PerThreadTimer(const char* const _region, const char* const _category)
  {}

  void start(void) const {}

  void stop(void) const {}

};

} // end namespace galois
#endif

