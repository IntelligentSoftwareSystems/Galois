/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#include "galois/Timer.h"
#include "galois/runtime/Statistics.h"

using namespace galois;

void Timer::start() { startT = clockTy::now(); }

void Timer::stop() { stopT = clockTy::now(); }

uint64_t Timer::get() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(stopT - startT)
      .count();
}

uint64_t Timer::get_usec() const {
  return std::chrono::duration_cast<std::chrono::microseconds>(stopT - startT)
      .count();
}

TimeAccumulator::TimeAccumulator() : ltimer(), acc(0) {}

void TimeAccumulator::start() { ltimer.start(); }

void TimeAccumulator::stop() {
  ltimer.stop();
  acc += ltimer.get_usec();
}

uint64_t TimeAccumulator::get() const { return acc / 1000; }
uint64_t TimeAccumulator::get_usec() const { return acc; }

TimeAccumulator& TimeAccumulator::operator+=(const TimeAccumulator& rhs) {
  acc += rhs.acc;
  return *this;
}

TimeAccumulator& TimeAccumulator::operator+=(const Timer& rhs) {
  acc += rhs.get_usec();
  return *this;
}


StatTimer::StatTimer(const char* const name, const char* const region) {
  const char *n = name ? name : "Time";
  const char *r = region ? region : "(NULL)";

  name_   = gstl::makeStr(n);
  region_ = gstl::makeStr(r);

  valid_ = false;
}

StatTimer::~StatTimer() {
  if (valid_) {
    stop();
  }

  // only report non-zero stat
  if (TimeAccumulator::get()) {
    galois::runtime::reportStat_Tmax(region_, name_, TimeAccumulator::get());
  }
}

void StatTimer::start() {
  TimeAccumulator::start();
  valid_ = true;
}

void StatTimer::stop() {
  valid_ = false;
  TimeAccumulator::stop();
}

uint64_t StatTimer::get_usec() const {
  return TimeAccumulator::get_usec();
}
