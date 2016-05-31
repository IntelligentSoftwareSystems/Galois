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
#include "Galois/gdeque.h"
#include "Galois/Substrate/PerThreadStorage.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Substrate/gio.h"
#include "Galois/Substrate/PaddedLock.h"
#include "Galois/Substrate/StaticInstance.h"
#include "Galois/Runtime/Mem.h"

#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/Serialize.h"

#include <cmath>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include <sstream>

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

  static uint32_t num_recv_expected;

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
      rStat->second.emplace_back(loop, category, numPagePoolAllocForThread(x));
    }
  }

  void addNumaAllocToStat(const std::string& loop, const std::string& category) {
    int nodes = Galois::Substrate::getThreadPool().getMaxNumaNodes();
    for (int x = 0; x < nodes; ++x) {
      auto rStat = Stats.getRemote(x);
      std::lock_guard<Substrate::SimpleLock> lg(rStat->first);
      //      rStat->second.emplace_back(loop, category, numNumaAllocForNode(x));
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

  static void printDistStats_landingPad(Galois::Runtime::RecvBuffer& buf){
    std::string recv_str;
    uint32_t from_ID;
    Galois::Runtime::gDeserialize(buf, from_ID, recv_str);
    Substrate::gPrint(recv_str);
    --num_recv_expected;
  }
  //Distributed version
  void printDistStats() {
    std::map<std::pair<std::string, std::string>, std::vector<unsigned long> > LKs;

    auto& net = Galois::Runtime::getSystemNetworkInterface();
    net.reportStats();

    num_recv_expected = net.Num - 1;
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
    std::stringstream ss;
    std::string str_ID = std::to_string(net.ID);
    ss << "[" + str_ID  + "]" + "STATTYPE,LOOP,CATEGORY,n,sum";
    for (unsigned x = 0; x <= maxThreadID; ++x)
      ss << ",T" + std::to_string(x);
    ss << "\n";
    //print all values
    for (auto ii = LKs.begin(), ee = LKs.end(); ii != ee; ++ii) {
      std::vector<unsigned long>& Values = ii->second;
      auto accum_val = std::accumulate(Values.begin(), Values.end(), static_cast<unsigned long>(0));
      ss <<  "[" + str_ID  + "]" + "STAT," + std::string(ii->first.first.c_str()) + "," + std::string(ii->first.second.c_str()) + "," + std::to_string(maxThreadID + 1) + "," + std::to_string(accum_val);

      for (unsigned x = 0; x <= maxThreadID; ++x)
        ss << "," + std::to_string(x < Values.size() ? Values.at(x) : 0);
      ss << "\n";

    }

    if(net.ID == 0){
      Substrate::gPrint(ss.str());

      while(num_recv_expected){
        net.handleReceives();
      }

    }
    else{
      //send to host 0 to print.
      Galois::Runtime::SendBuffer b;
      gSerialize(b, net.ID, ss.str());
      net.send(0, printDistStats_landingPad, b);
      net.flush();
    }

    ss.str(std::string());
    ss.clear();
  }

  void printDistStatsGlobal(std::string type, std::string val){
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    if(net.ID == 0){
      Substrate::gPrint(type, " : ", val, "\n\n");
    }
  }

};

uint32_t StatManager::num_recv_expected;

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

void Galois::Runtime::reportStatGlobal(const std::string& type, const std::string& val) {
  SM.get()->printDistStatsGlobal(type, val);
}
void Galois::Runtime::reportStatGlobal(const std::string& type, unsigned long val) {
  SM.get()->printDistStatsGlobal(type, std::to_string(val));
}

void Galois::Runtime::printStats() {
  SM.get()->printStats();
}

void Galois::Runtime::printDistStats() {
  SM.get()->printDistStats();
}

void Galois::Runtime::reportPageAlloc(const char* category) {
  SM.get()->addPageAllocToStat(std::string("(NULL)"), std::string(category ? category : "(NULL)"));
}

void Galois::Runtime::reportNumaAlloc(const char* category) {
  SM.get()->addNumaAllocToStat(std::string("(NULL)"), std::string(category ? category : "(NULL)"));
}
