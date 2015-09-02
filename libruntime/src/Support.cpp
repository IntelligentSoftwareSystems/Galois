/** Support functions -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a gramework to exploit
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
#include "Galois/gdeque.h"
#include "Galois/Substrate/PerThreadStorage.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Substrate/gio.h"
#include "Galois/Substrate/PaddedLock.h"
#include "Galois/Substrate/StaticInstance.h"
#include "Galois/Runtime/Mem.h"

#include <cmath>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace Galois {
namespace Runtime {
extern unsigned activeThreads;
} } //end namespaces

using namespace Galois;
using namespace Galois::Runtime;

namespace {

class StatManager {
  typedef std::tuple<std::string, std::string, unsigned long> RecordTy;

  Galois::Substrate::PerThreadStorage<std::pair<Substrate::SimpleLock, gdeque<RecordTy> > > Stats;

public:
  StatManager() {}

  void addToStat(const std::string& loop, const std::string& category, size_t value) {
    auto* lStat = Stats.getLocal();
    // lStat->first.lock();
    std::lock_guard<Substrate::SimpleLock> lg(lStat->first);
    lStat->second.emplace_back(loop, category, value);
    // lStat->first.unlock();
  }

  void addToStat(Galois::Statistic* value) {
    for (unsigned x = 0; x < Galois::Runtime::activeThreads; ++x) {
      auto rStat = Stats.getRemote(x);
      std::lock_guard<Substrate::SimpleLock> lg(rStat->first);
      rStat->second.emplace_back(value->getLoopname(), value->getStatname(), value->getValue(x));
    }
  }

  void addPageAllocToStat(const std::string& loop, const std::string& category) {
    for (unsigned x = 0; x < Galois::Runtime::activeThreads; ++x) {
      auto rStat = Stats.getRemote(x);
      std::lock_guard<Substrate::SimpleLock> lg(rStat->first);
      rStat->second.emplace_back(loop, category, numPageAllocForThread(x));
    }
  }

  void addNumaAllocToStat(const std::string& loop, const std::string& category) {
    int nodes = Galois::Runtime::numNumaNodes();
    for (int x = 0; x < nodes; ++x) {
      auto rStat = Stats.getRemote(x);
      std::lock_guard<Substrate::SimpleLock> lg(rStat->first);
      rStat->second.emplace_back(loop, category, numNumaAllocForNode(x));
    }
  }

  //Assume called serially
  void printStats() {
    std::map<std::pair<std::string, std::string>, std::vector<unsigned long> > LKs;
    
    unsigned maxThreadID = 0;
    //Find all loops and keys
    for (unsigned x = 0; x < Stats.size(); ++x) {
      auto rStat = Stats.getRemote(x);
      std::lock_guard<Substrate::SimpleLock> lg(rStat->first);
      for (auto ii = rStat->second.begin(), ee = rStat->second.end();
	   ii != ee; ++ii) {
        maxThreadID = x;
	auto& v = LKs[std::make_pair(std::get<0>(*ii), std::get<1>(*ii))];
        if (v.size() <= x)
          v.resize(x+1);
        v[x] += std::get<2>(*ii);
      }
    }
    //print header
    Substrate::gPrint("STATTYPE,LOOP,CATEGORY,n,sum");
    for (unsigned x = 0; x <= maxThreadID; ++x)
      Substrate::gPrint(",T", x);
    Substrate::gPrint("\n");
    //print all values
    for (auto ii = LKs.begin(), ee = LKs.end(); ii != ee; ++ii) {
      std::vector<unsigned long>& Values = ii->second;
      Substrate::gPrint("STAT,",
                 ii->first.first.c_str(), ",",
                 ii->first.second.c_str(), ",",
                 maxThreadID + 1, ",",
                 std::accumulate(Values.begin(), Values.end(), static_cast<unsigned long>(0))
                 );
      for (unsigned x = 0; x <= maxThreadID; ++x)
        Substrate::gPrint(",", x < Values.size() ? Values.at(x) : 0);
      Substrate::gPrint("\n");
    }
  }
};

static Substrate::StaticInstance<StatManager> SM;

}

void Galois::Runtime::reportStat(const char* loopname, const char* category, unsigned long value) {
  SM.get()->addToStat(std::string(loopname ? loopname : "(NULL)"), 
		      std::string(category ? category : "(NULL)"),
		      value);
}

void Galois::Runtime::reportStat(const std::string& loopname, const std::string& category, unsigned long value) {
  SM.get()->addToStat(loopname, category, value);
}

void Galois::Runtime::reportStat(Galois::Statistic* value) {
  SM.get()->addToStat(value);
}

void Galois::Runtime::reportStatGlobal(const std::string&, const std::string&) {
}
void Galois::Runtime::reportStatGlobal(const std::string&, unsigned long) {
}

void Galois::Runtime::printStats() {
  SM.get()->printStats();
}

void Galois::Runtime::reportPageAlloc(const char* category) {
  SM.get()->addPageAllocToStat(std::string("(NULL)"), std::string(category ? category : "(NULL)"));
}

void Galois::Runtime::reportNumaAlloc(const char* category) {
  SM.get()->addNumaAllocToStat(std::string("(NULL)"), std::string(category ? category : "(NULL)"));
}
