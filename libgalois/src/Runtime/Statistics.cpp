/** Statistic interface -*- C++ -*-
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
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/GaloisConfig.h"
#include "Galois/Runtime/Statistics.h"
#include "Galois/Runtime/StatCollector.h"

#include <iostream>
#include <fstream>

void Galois::Runtime::reportLoopInstance(const std::string& loopname) {
  getGaloisConfig().getStat().beginLoopInstance(loopname);
}

void Galois::Runtime::reportStat(const std::string& loopname, const std::string& category, unsigned long value, unsigned TID) {
  getGaloisConfig().getStat().addToStat(loopname, category, value, TID, 0);
}

void Galois::Runtime::reportStat(const std::string& loopname, const std::string& category, const std::string& value, unsigned TID) {
  getGaloisConfig().getStat().addToStat(loopname, category, value, TID, 0);
}

void Galois::Runtime::reportStatGlobal(const std::string& category, const std::string& value) {
  getGaloisConfig().getStat().addToStat(std::string("(Global)"), category, value, 0, 0);
}

void Galois::Runtime::reportStatGlobal(const std::string& category, unsigned long value) {
  getGaloisConfig().getStat().addToStat(std::string("(Global)"), category, value, 0, 0);
}


//! Prints all stats
void Galois::Runtime::printStats() {
  auto& cfg = getGaloisConfig();
  std::string loc = cfg.getStatLoc();
  bool R = cfg.getStatInR();
  bool json = cfg.getStatJSON();
  if (loc.size()) {
    if (loc == "-") {
      if (R)
        cfg.getStat().printStatsForR(std::cout, json);
      else
        cfg.getStat().printStats(std::cout);
    } else {
      std::ofstream file(loc);
      if (R)
        cfg.getStat().printStatsForR(file, json);
      else
        cfg.getStat().printStats(file);
    }
  }
}

void Galois::Runtime::setStatOutput(const std::string& loc) {
  getGaloisConfig().setStatLoc(loc);
}

void Galois::Runtime::setStatFormat(bool R, bool json) {
  getGaloisConfig().setStatInR(R);
  getGaloisConfig().setStatJSON(json);
}
