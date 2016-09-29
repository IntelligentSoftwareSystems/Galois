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
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_STATTIMER_H
#define GALOIS_STATTIMER_H

#include "Galois/Timer.h"
#include "Galois/Statistic.h"

namespace Galois {

//! Provides statistic interface around timer
class StatTimer : public TimeAccumulator {
  std::string statname;
  std::string loopname;
  bool valid;

protected:
  StatTimer(const std::string& n, const std::string& l, bool s)
    :statname(n), loopname(l), valid(false) {
    if (s)
      start();
  }

public:
  StatTimer(const std::string& n, const std::string& l = "(Global)")
    :StatTimer(n, l, false) {}
  StatTimer(const std::string& n, start_now_t t)
    :StatTimer(n, "(Global)", true) {}
  StatTimer(const std::string& n, const std::string& l, start_now_t t)
    :StatTimer(n, l, true) {}
  StatTimer()
    :StatTimer("Time", "(Global)", false) {}
  StatTimer(start_now_t t)
    :StatTimer("Time", "(Global)", true) {}

  ~StatTimer() {
    if (valid)
      stop();
    if (TimeAccumulator::get()) // only report non-zero stat
      Runtime::reportStat(loopname, statname, get(), Runtime::ThreadPool::getTID());
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


} // namespace Galois
#endif
