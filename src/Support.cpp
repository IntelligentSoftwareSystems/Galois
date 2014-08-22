/** Support functions -*- C++ -*-
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
#include "Galois/Statistic.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Support.h"
#include "Galois/gdeque.h"
#include "Galois/Runtime/ll/StaticInstance.h"
#include "Galois/Runtime/ll/PaddedLock.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/mm/Mem.h"

#include <set>
#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <mutex>
#include <numeric>

using namespace Galois;
using namespace Galois::Runtime;

namespace {

class StatManager {
  typedef std::tuple<std::string, std::string, unsigned long> RecordTy;

  Galois::Runtime::PerThreadStorage<std::pair<LL::SimpleLock, gdeque<RecordTy> > > Stats;

public:
  StatManager() {}

  void addToStat(const std::string& loop, const std::string& category, size_t value) {
    auto* lStat = Stats.getLocal();
    // lStat->first.lock();
    std::lock_guard<LL::SimpleLock> lg(lStat->first);
    lStat->second.emplace_back(loop, category, value);
    // lStat->first.unlock();
  }

  void addToStat(Galois::Statistic* value) {
    for (unsigned x = 0; x < Galois::Runtime::activeThreads; ++x) {
      auto rStat = Stats.getRemote(x);
      std::lock_guard<LL::SimpleLock> lg(rStat->first);
      rStat->second.emplace_back(value->getLoopname(), value->getStatname(), value->getValue(x));
    }
  }

  void addPageAllocToStat(const std::string& loop, const std::string& category) {
    for (unsigned x = 0; x < Galois::Runtime::activeThreads; ++x) {
      auto rStat = Stats.getRemote(x);
      std::lock_guard<LL::SimpleLock> lg(rStat->first);
      rStat->second.emplace_back(loop, category, MM::numPageAllocForThread(x));
    }
  }

  void addNumaAllocToStat(const std::string& loop, const std::string& category) {
    int nodes = Galois::Runtime::MM::numNumaNodes();
    for (int x = 0; x < nodes; ++x) {
      auto rStat = Stats.getRemote(x);
      std::lock_guard<LL::SimpleLock> lg(rStat->first);
      rStat->second.emplace_back(loop, category, MM::numNumaAllocForNode(x));
    }
  }

  //Assume called serially
  void printStats() {
    std::map<std::pair<std::string, std::string>, std::vector<unsigned long> > LKs;
    
    unsigned maxThreadID = 0;
    //Find all loops and keys
    for (unsigned x = 0; x < Stats.size(); ++x) {
      auto rStat = Stats.getRemote(x);
      std::lock_guard<LL::SimpleLock> lg(rStat->first);
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
    LL::gPrint("STATTYPE,LOOP,CATEGORY,n,sum");
    for (unsigned x = 0; x <= maxThreadID; ++x)
      LL::gPrint(",T", x);
    LL::gPrint("\n");
    //print all values
    for (auto ii = LKs.begin(), ee = LKs.end(); ii != ee; ++ii) {
      std::vector<unsigned long>& Values = ii->second;
      LL::gPrint("STAT,",
                 ii->first.first.c_str(), ",",
                 ii->first.second.c_str(), ",",
                 maxThreadID + 1, ",",
                 std::accumulate(Values.begin(), Values.end(), static_cast<unsigned long>(0))
                 );
      for (unsigned x = 0; x <= maxThreadID; ++x)
        LL::gPrint(",", x < Values.size() ? Values.at(x) : 0);
      LL::gPrint("\n");
    }
  }
};

static LL::StaticInstance<StatManager> SM;

}

bool Galois::Runtime::inGaloisForEach = false;

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

void Galois::Runtime::printStats() {
  SM.get()->printStats();
}

void Galois::Runtime::reportPageAlloc(const char* category) {
  SM.get()->addPageAllocToStat(std::string("(NULL)"), std::string(category ? category : "(NULL)"));
}

void Galois::Runtime::reportNumaAlloc(const char* category) {
  SM.get()->addNumaAllocToStat(std::string("(NULL)"), std::string(category ? category : "(NULL)"));
}
