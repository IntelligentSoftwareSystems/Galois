/** Simple timer support -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
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
 * reser
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "galois/Timer.h"

using namespace galois;

void Timer::start() {
  startT = clockTy::now();
}

void Timer::stop() {
  stopT = clockTy::now();
}

unsigned long Timer::get() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(stopT-startT).count();
}

unsigned long Timer::get_usec() const {
  return std::chrono::duration_cast<std::chrono::microseconds>(stopT-startT).count();
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
