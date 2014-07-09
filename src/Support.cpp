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
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/mm/Mem.h"
#include "Galois/Runtime/ll/StaticInstance.h"
#include "Galois/Runtime/ll/PaddedLock.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/mm/Mem.h"
#include "Galois/gdeque.h"
#include "Galois/Statistic.h"

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
  // loop, category, hostid, thread id, value
  typedef std::tuple<std::string, std::string, uint32_t, uint32_t, size_t> RecordTy;

  Galois::Runtime::PerThreadStorage<std::pair<LL::SimpleLock, gdeque<RecordTy> > > Stats;

public:
  StatManager() {}

  void addToStat(const std::string& loop, const std::string& category, uint32_t host, uint32_t tid, size_t value) {
    auto lStat = Stats.getLocal();
    std::lock_guard<LL::SimpleLock> lg(lStat->first);
    lStat->second.emplace_back(loop, category, host, tid, value);
  }

  //Assume called serially
  void printStats() {
    // (loop, category) -> host -> tid -> value
    std::vector<uint32_t> maxTids;
    std::map<std::pair<std::string, std::string>, std::vector<std::vector<size_t>>> LKs;
    
    //Find all loops and keys
    for (unsigned x = 0; x < Stats.size(); ++x) {
      auto rStat = Stats.getRemote(x);
      std::lock_guard<LL::SimpleLock> lg(rStat->first);
      for (auto& record : rStat->second) {
        uint32_t host = std::get<2>(record);
        uint32_t tid = std::get<3>(record);
        size_t value = std::get<4>(record);

        if (maxTids.size() <= host)
          maxTids.resize(host + 1);
        auto& maxTid = maxTids[host];
        maxTid = std::max(maxTid, tid);

	auto& v = LKs[std::make_pair(std::get<0>(record), std::get<1>(record))];
        if (v.size() <= host)
          v.resize(host + 1);
        auto& vv = v[host];
        if (vv.size() <= tid)
          vv.resize(tid + 1);
        vv[tid] += value;
      }
    }
    //print header
    LL::gPrint("STATTYPE,LOOP,CATEGORY,n,sum");
    size_t total = 0;
    for (uint32_t host = 0; host < maxTids.size(); ++host) {
      for (uint32_t tid = 0; tid < maxTids[host]; ++tid) {
        LL::gPrint(",H", host, "T", tid);
        total += 1;
      }
    }
    LL::gPrint("\n");
    //print all values
    for (auto ii = LKs.begin(), ee = LKs.end(); ii != ee; ++ii) {
      auto& v = ii->second;

      auto& values = ii->second;
      size_t accum = 0;
      for (auto& x : values)
        accum += std::accumulate(x.begin(), x.end(), 0);

      LL::gPrint("STAT,",
                 ii->first.first, ",",
                 ii->first.second, ",",
                 total, ",",
                 accum
                 );
      for (uint32_t host = 0; host < maxTids.size(); ++host) {
        for (uint32_t tid = 0; tid < maxTids[host]; ++tid) {
          size_t value = 0;
          if (host < v.size() && tid < v[host].size())
            value = v[host][tid];
          LL::gPrint(",", value);
        }
      }
      LL::gPrint("\n");
    }
  }
};

static LL::StaticInstance<StatManager> SM;

static void reportStatInternal(std::string loop, std::string category, uint32_t host, uint32_t tid, size_t value) {
  if (Galois::Runtime::NetworkInterface::ID == 0)
    SM.get()->addToStat(loop, category, host, tid, value);
  else
    Galois::Runtime::getSystemNetworkInterface().sendAlt(0, reportStatInternal, loop, category, host, tid, value);
}

}

bool Galois::Runtime::inGaloisForEach = false;

void Galois::Runtime::reportPageAlloc(const char* category) {
  // TODO ask other hosts too
  for (unsigned x = 0; x < Galois::Runtime::activeThreads; ++x)
      Galois::Runtime::reportStat(nullptr, category, Galois::Runtime::MM::numPageAllocForThread(x));
}

void Galois::Runtime::reportNumaAlloc(const char* category) {
  // TODO ask other hosts too
  int nodes = Galois::Runtime::MM::numNumaNodes();
   for (int x = 0; x < nodes; ++x)
     Galois::Runtime::reportStat(nullptr, category, Galois::Runtime::MM::numNumaAllocForNode(x));
}

void Galois::Runtime::reportStat(const char* loopname, const char* category, unsigned long value) {
  reportStatInternal(std::string(loopname ? loopname : "(NULL)"), 
		     std::string(category ? category : "(NULL)"),
		     NetworkInterface::ID, LL::getTID(), value);
}

void Galois::Runtime::reportStat(const std::string& loopname, const std::string& category, unsigned long value) {
  reportStatInternal(loopname, category, NetworkInterface::ID, LL::getTID(), value);
}

void Galois::Runtime::reportStat(Galois::Statistic* value) {
  for (unsigned x = 0; x < activeThreads; ++x)
    reportStatInternal(value->getLoopname(), value->getStatname(), NetworkInterface::ID, x, value->getValue(x));
}

void Galois::Runtime::printStats() {
  SM.get()->printStats();
}
