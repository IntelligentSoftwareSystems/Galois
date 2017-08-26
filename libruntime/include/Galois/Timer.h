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

#include <chrono>

#include <cstdio>
#include <ctime>

namespace Galois {

//! Flag type for {@link StatTimer}
struct start_now_t {};
constexpr start_now_t start_now = start_now_t();

//! A simple timer
class Timer {
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
class TimeAccumulator {
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
class ThreadTimer {
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


}
#endif

