/** Statistics functions -*- C++ -*-
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

#include "Galois/Runtime/ll/StaticInstance.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/ll/SimpleLock.h"
#include "Galois/Runtime/Network.h"
#include "Galois/Statistic.h"

#include <map>
#include <vector>
#include <numeric>
#include <mutex>

using Galois::Runtime::LL::gPrint;
using Galois::Runtime::LL::SimpleLock;

namespace {

class StatCollector {

  typedef std::pair<std::string, std::string> KeyTy;

  //<Loop, Name> -> hostid -> threads id -> stat
  std::map<KeyTy, std::map<uint32_t, std::map<uint32_t, unsigned long> > > Stats;

  uint32_t maxTID;
  uint32_t maxHost;

  SimpleLock<true> StatLock;

  KeyTy mkKey(const std::string& loop, const std::string& category) {
    return std::make_pair(loop,category);
  }

  std::vector<unsigned long> gather(const KeyTy& k) {
    std::vector<unsigned long> v;
    for (unsigned x = 0; x <= maxHost; ++x)
      for (unsigned y = 0; y <= maxTID; ++y)
	v.push_back(Stats[k][x][y]);
    return v;
  }

public:

  StatCollector() :maxTID(0), maxHost(0) {}

  void addToStat(const std::string& loop, const std::string& category, uint32_t host, uint32_t tid, size_t value) {
    std::lock_guard<SimpleLock<true>> lock(StatLock);
    Stats[mkKey(loop, category)][host][tid] += value;
    if (maxHost < host) maxHost = host;
    if (maxTID < tid) maxTID = tid;
  }

  void printStats() {
    std::lock_guard<SimpleLock<true>> lock(StatLock);
    //print header
    gPrint("STATTYPE,LOOP,CATEGORY,n,sum");
    for (unsigned x = 0; x <= maxHost; ++x)
      for (unsigned y = 0; y <= maxTID; ++y)
	gPrint(",T", y, "H", x);
    gPrint("\n");
    //print all values
    for(auto ii = Stats.begin(), ee = Stats.end(); ii != ee; ++ii) {
      std::vector<unsigned long> Values = gather(ii->first);
      gPrint("STAT,",
	     ii->first.first.c_str(), ",",
	     ii->first.second.c_str(), ",",
	     (maxTID+1)*(maxHost+1), ",",
	     std::accumulate(Values.begin(), Values.end(), 0)
	     );
      for (unsigned x = 0; x <= maxHost; ++x)
	for (unsigned y = 0; y <= maxTID; ++y)
	  gPrint(",", Values[x*(maxTID+1)+y]);
      gPrint("\n");
    }
  }
};

static Galois::Runtime::LL::StaticInstance<StatCollector> SM;

static void reportStatInternal(const std::string loop, const std::string category, uint32_t host, uint32_t tid, size_t value) {
  if (Galois::Runtime::NetworkInterface::ID == 0)
    SM.get()->addToStat(loop, category, host, tid, value);
  else
    Galois::Runtime::getSystemNetworkInterface().sendAlt(0, reportStatInternal, loop, category, host, tid, value);
}

}

void Galois::Runtime::reportStat(const char* loopname, const char* category, unsigned long value) {
  reportStatInternal(std::string(loopname ? loopname : "(NULL)"), 
		     std::string(category ? category : "(NULL)"),
		     NetworkInterface::ID, LL::getTID(), value);
}

void Galois::Runtime::reportStat(const std::string& loopname, const std::string& category, unsigned long value) {
  reportStatInternal(loopname, category, NetworkInterface::ID,  LL::getTID(), value);
}

void Galois::Runtime::reportStat(Galois::Statistic* value) {
  for (unsigned x = 0; x < activeThreads; ++x)
    reportStatInternal(value->getLoopname(), value->getStatname(), NetworkInterface::ID, x, value->getValue(x));
}

void Galois::Runtime::printStats() {
  SM.get()->printStats();
}
