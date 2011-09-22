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

#include "Accumulator.h"
#include "Runtime/Support.h"
#include "Timer.h"

namespace Galois {

template<typename T>
class Statistic : public GAccumulator<T> {
  const char* name;
public:
  Statistic(const char* _name) :name(_name) {}
  ~Statistic() {
    GaloisRuntime::reportStatSum(name, GAccumulator<T>::get());
  }
};

class StatTimer : public Timer {
  const char* name;
  const char* loopname;
public:
  StatTimer(const char* n = "Time", const char* l = 0) :name(n), loopname(l) {}
  ~StatTimer() {
    GaloisRuntime::reportStatSum(name, get(), loopname);
  }

  void start() {
    Timer::start();
  }

  void stop() {
    Timer::stop();
  }
};

}

#endif
