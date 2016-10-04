/** Galois user interface -*- C++ -*-
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

#ifndef GALOIS_RUNTIME_GALOISCONFIG_H
#define GALOIS_RUNTIME_GALOISCONFIG_H

#include "Galois/Runtime/StatCollector.h"
#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/Termination.h"

namespace Galois {
namespace Runtime {

class GaloisConfig {
  
  unsigned activeThreads;
  StatCollector stat;
  bool stat_R, stat_json;
  std::string stat_file;
  std::map<unsigned, std::unique_ptr<Barrier> > barrierMap;
  std::unique_ptr<TerminationDetection> term;

public:
  GaloisConfig() : activeThreads(1), stat_R(false), stat_json(false), term(createTermination()) {}

  unsigned getActiveThreads() const noexcept {
    return activeThreads;
  }
  unsigned setActiveThreads(unsigned num) noexcept {
    activeThreads = num;
    return activeThreads;
  }

  StatCollector& getStat() noexcept {
    return stat;
  }

  void setStatInR(bool R) { stat_R = R; }
  bool getStatInR() { return stat_R; }
  void setStatJSON(bool j) { stat_json = j; }
  bool getStatJSON() { return stat_json; }
  void setStatLoc(const std::string& s) { stat_file = s; }
  std::string getStatLoc() { return stat_file; }

  Barrier& getBarrier(unsigned num) {
    auto& b = barrierMap[num];
    if (!b)
      b = createBarrier(num);
    return *b;
  }

  TerminationDetection& getTermination(unsigned num) {
    term->init(num);
    return *term;
  }

};

GaloisConfig& getGaloisConfig();

} // namespace Runtime
} // namespace Galois

#endif
