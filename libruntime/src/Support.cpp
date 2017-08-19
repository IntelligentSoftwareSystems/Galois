/** Support functions -*- C++ -*-
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
 * reserved.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Statistic.h"
#include "Galois/Runtime/StatCollector.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Substrate/gio.h"
#include "Galois/Substrate/StaticInstance.h"
#include "Galois/Runtime/Mem.h"
#include <iostream>

#include <cmath>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

namespace Galois {
namespace Runtime {
extern unsigned activeThreads;
} } //end namespaces

using namespace Galois;
using namespace Galois::Runtime;

static Substrate::StaticInstance<Galois::Runtime::StatCollector> SM;

#if 0
void Galois::Runtime::reportLoopInstance(const char* loopname) {
  SM.get()->beginLoopInstance(std::string(loopname ? loopname : "(NULL)"));
}

//FIXME: fix on Networked
void Galois::Runtime::reportStat(const char* loopname, const char* category, unsigned long value, unsigned TID) {
  SM.get()->addToStat(std::string(loopname ? loopname : "(NULL)"), 
		      std::string(category ? category : "(NULL)"),
		      value, TID, 0);
}
//FIXME: fix on Networked
void Galois::Runtime::reportStat(const char* loopname, const char* category, const std::string& value, unsigned TID) {
  SM.get()->addToStat(std::string(loopname ? loopname : "(NULL)"), 
		      std::string(category ? category : "(NULL)"),
		      value, TID, 0);
}

//FIXME: fix on Networked
void Galois::Runtime::reportStat(const std::string& loopname, const std::string& category, unsigned long value, unsigned TID) {
  SM.get()->addToStat(loopname, category, value, TID, 0);
}

//FIXME: fix on Networked
void Galois::Runtime::reportStat(const std::string& loopname, const std::string& category, const std::string& value, unsigned TID) {
  SM.get()->addToStat(loopname, category, value, TID, 0);
}
#endif

void Galois::Runtime::reportStatGlobal(const std::string&, const std::string&) {
}
void Galois::Runtime::reportStatGlobal(const std::string&, unsigned long) {
}

static std::ofstream& openIfNot_output(std::string fname) {
  static std::ofstream output_file;
  if(!output_file.is_open()){
    output_file.open(fname, std::ios_base::app);
  }
  assert(output_file.is_open());
  return output_file;
}

#if 0
void Galois::Runtime::printStats() {
    // SM.get()->printDistStats(std::cout);
    SM.get()->printStats(std::cout);
  //SM.get()->printStatsForR(std::cout, false);
  //  SM.get()->printStatsForR(std::cout, true);
}

void Galois::Runtime::printStats(std::string fname) {
  if(fname == "")
    SM.get()->printStatsForR(std::cout, false);
  else{
    auto& out = openIfNot_output(fname);
    SM.get()->printStatsForR(out, false);
    out.close();
  }
}
#endif

void Galois::Runtime::reportPageAlloc(const char* category) {
  for (unsigned x = 0; x < Galois::Runtime::activeThreads; ++x)
    reportStat("(NULL)", category, static_cast<unsigned long>(numPagePoolAllocForThread(x)), x);
}

void Galois::Runtime::reportNumaAlloc(const char* category) {
  int nodes = Substrate::ThreadPool::getThreadPool().getMaxNumaNodes();
  for (int x = 0; x < nodes; ++x) {
    //auto rStat = Stats.getRemote(x);
    //std::lock_guard<Substrate::SimpleLock> lg(rStat->first);
    //      rStat->second.emplace_back(loop, category, numNumaAllocForNode(x));
  }
  //  SM.get()->addNumaAllocToStat(std::string("(NULL)"), std::string(category ? category : "(NULL)"));
}
