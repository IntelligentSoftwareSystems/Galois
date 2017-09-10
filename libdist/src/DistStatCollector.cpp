/** StatCollector Implementation -*- C++ -*-
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

// TODO: this file was copied over from libruntime/src/StatCollector.cpp. 
// TODO: remove the duplicated code after inheriting from Galois::Runtime::StatCollector

#include "Galois/Runtime/DistStatCollector.h"

#include <cmath>
#include <new>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace Galois;
using namespace Galois::Runtime;

DistStatCollector::DistStatCollector(const std::string& outfile): StatCollector(outfile) {}


static void recvAtHost_0_int(uint32_t HostID, const std::string loopname, const std::string category, const size_t value, unsigned TID) {
  reportStatDist(loopname, category, value, TID, HostID);
}

static void recvAtHost_0_double(uint32_t HostID, const std::string loopname, const std::string category, const double value, unsigned TID) {
  reportStatDist(loopname, category, value, TID, HostID);
}

static void recvAtHost_0_str(uint32_t HostID, const std::string loopname, const std::string category, const std::string value, unsigned TID) {
  reportStatDist(loopname, category, value, TID, HostID);
}

void DistStatCollector::combineAtHost_0(void) {
  Galois::Runtime::getHostBarrier().wait();

  if (getHostID() != 0) {
    for (auto& p : Stats) {
      const auto tid = std::get<1>(p.first);
      const auto& loopname = *std::get<2>(p.first);
      const auto& category = *std::get<3>(p.first);


      switch (p.second.mode) {

        case StatCollector::RecordTy::INT: 
          {
            size_t val = p.second.intVal();
            getSystemNetworkInterface().sendSimple(0, recvAtHost_0_int, loopname, category, val, tid);
            break;
          }

        case StatCollector::RecordTy::DOUBLE: 
          {
            double val = p.second.doubleVal();
            getSystemNetworkInterface().sendSimple(0, recvAtHost_0_double, loopname, category, val, tid);
            break;
          }

        case StatCollector::RecordTy::STR: 
          {
            const std::string& val = p.second.strVal();
            getSystemNetworkInterface().sendSimple(0, recvAtHost_0_str, loopname, category, val, tid);
            break;
          }
        default:
          std::abort();
          break;
      }

    }
  }

  Galois::Runtime::getHostBarrier().wait();
}


void DistStatCollector::printStats(void) {

  combineAtHost_0();

  if (getHostID() != 0)  {
    return;

  } else {
    StatCollector::printStats();
  }

}

