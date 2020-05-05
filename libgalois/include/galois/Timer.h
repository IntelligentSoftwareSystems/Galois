/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef GALOIS_TIMER_H
#define GALOIS_TIMER_H

#include <chrono>

#include "galois/config.h"
#include "galois/gstl.h"

namespace galois {

//! A simple timer
class Timer {
  typedef std::chrono::steady_clock clockTy;
  // typedef std::chrono::high_resolution_clock clockTy;
  std::chrono::time_point<clockTy> startT, stopT;

public:
  void start();
  void stop();
  uint64_t get() const;
  uint64_t get_usec() const;
};

//! A multi-start time accumulator.
//! Gives the final runtime for a series of intervals
class TimeAccumulator {
  Timer ltimer;
  uint64_t acc;

public:
  TimeAccumulator();

  void start();
  //! adds the current timed interval to the total
  void stop();
  uint64_t get() const;
  uint64_t get_usec() const;
  TimeAccumulator& operator+=(const TimeAccumulator& rhs);
  TimeAccumulator& operator+=(const Timer& rhs);
};

//! Galois Timer that automatically reports stats upon destruction
//! Provides statistic interface around timer
class StatTimer : public TimeAccumulator {
  gstl::Str name_;
  gstl::Str region_;
  bool valid_;

public:
  StatTimer(const char* name, const char* region);

  StatTimer(const char* const n) : StatTimer(n, nullptr) {}

  StatTimer() : StatTimer(nullptr, nullptr) {}

  StatTimer(const StatTimer&) = delete;
  StatTimer(StatTimer&&)      = delete;
  StatTimer& operator=(const StatTimer&) = delete;
  StatTimer& operator=(StatTimer&&) = delete;

  ~StatTimer();

  void start();
  void stop();
  uint64_t get_usec() const;
};

template <bool Enable>
class CondStatTimer : public StatTimer {
public:
  CondStatTimer(const char* const n, const char* region)
      : StatTimer(n, region) {}

  CondStatTimer(const char* region) : CondStatTimer("Time", region) {}
};

template <>
class CondStatTimer<false> {
public:
  CondStatTimer(const char*) {}
  CondStatTimer(const char* const, const char*) {}

  void start() const {}
  void stop() const {}
  uint64_t get_usec() const { return 0; }
};

template <typename F>
void timeThis(const F& f, const char* const name) {
  StatTimer t("Time", name);

  t.start();

  f();

  t.stop();
}

} // end namespace galois
#endif
