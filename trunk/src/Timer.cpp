/** Simple timer support -*- C++ -*-
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
#include "Galois/Timer.h"

// This is linux/bsd specific
#include <sys/time.h>

using namespace Galois;

Timer::Timer()
  :_start_hi(0), _start_low(0), _stop_hi(0), _stop_low(0)
{}

void Timer::start() {
  timeval start;
  gettimeofday(&start, 0);
  _start_hi = start.tv_sec;
  _start_low = start.tv_usec;
}

void Timer::stop() {
  timeval stop;
  gettimeofday(&stop, 0);
  _stop_hi = stop.tv_sec;
  _stop_low = stop.tv_usec;
}

unsigned long Timer::get() const {
  unsigned long msec = _stop_hi - _start_hi;
  msec *= 1000;
  if (_stop_low > _start_low)
    msec += (_stop_low - _start_low) / 1000;
  else {
    msec -= 1000; //borrow
    msec += (_stop_low + 1000000 - _start_low) / 1000;
  }
  return msec;
}

unsigned long Timer::get_usec() const {
  unsigned long usec = _stop_hi - _start_hi;
  usec *= 1000000;
  if (_stop_low > _start_low)
    usec += (_stop_low - _start_low);
  else {
    usec -= 1000000; //borrow
    usec += (_stop_low + 1000000 - _start_low);
  }
  return usec;
}

TimeAccumulator::TimeAccumulator()
  :ltimer(), acc(0)
{}

void TimeAccumulator::start() {
  ltimer.start();
}

void TimeAccumulator::stop() {
  ltimer.stop();
  acc += ltimer.get_usec();
}

unsigned long TimeAccumulator::get() const {
  return acc / 1000;
}

TimeAccumulator& TimeAccumulator::operator+=(const TimeAccumulator& rhs) {
  acc += rhs.acc;
  return *this;
}

TimeAccumulator& TimeAccumulator::operator+=(const Timer& rhs) {
  acc += rhs.get_usec();
  return *this;
}

