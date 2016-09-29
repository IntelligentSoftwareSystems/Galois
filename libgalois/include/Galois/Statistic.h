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

#include "Galois/Runtime/PerThreadStorage.h"
//#include "Galois/Runtime/Support.h"
//#include "Galois/Runtime/Sampling.h"
//#include "Galois/Timer.h"

//#include "boost/utility.hpp"

//#include <deque>
#include <string>

namespace Galois {

/**
 * Basic per-thread statistics counter.
 */
template<typename Ty>
class StatisticBase {
  std::string statname;
  std::string loopname;
  Runtime::PerThreadStorage<std::pair<bool, Ty> > vals;

public:
  StatisticBase(const std::string& _sn, std::string _ln = "(Global)", Ty init = Ty())
    : statname(_sn), loopname(_ln), vals(false,init) { }
  
  ~StatisticBase() {
    report();
  }

  //! Adds stat to stat pool, usually deconsructor or StatManager calls this for you.
  void report() {
    for (unsigned x = 0; x < vals.size(); ++x)  {
      auto* ptr = vals.get(x);
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
};

using Statistic = StatisticBase<unsigned long>;

// /**
//  * Controls lifetime of stats. Users usually instantiate in main to print out
//  * statistics at program exit.
//  */
// class StatManager: private boost::noncopyable {
//   std::deque<Statistic*> stats;

// public:
//   ~StatManager() {
//     for(auto* s : stats)
//       s->report();
//     Galois::Runtime::printStats();
//   }
//   //! Statistics that are not lexically scoped must be added explicitly
//   void push(Statistic& s) {
//     stats.push_back(&s);
//   }
// };

} // namespace Galois
#endif
