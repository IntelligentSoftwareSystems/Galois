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

#include "Galois/Runtime/StatCollector.h"

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

namespace Galois {
namespace Runtime {
extern unsigned activeThreads;
} } //end namespaces

using namespace Galois;
using namespace Galois::Runtime;

static Galois::Runtime::StatCollector* SC;

void Galois::Runtime::internal::setStatCollector(Galois::Runtime::StatCollector* sc) {
  GALOIS_ASSERT(!(SC && sc), "StatCollector.cpp: Double Initialization of SC");
  SC = sc;
}

StatCollector::StatCollector(const std::string& outfile): m_outfile(outfile) {}

StatCollector::~StatCollector() {
  printStats();
}

const std::string* Galois::Runtime::StatCollector::getSymbol(const std::string& str) const {
  auto ii = symbols.find(str);
  if (ii == symbols.cend())
    return nullptr;
  return &*ii;
}

const std::string* Galois::Runtime::StatCollector::getOrInsertSymbol(const std::string& str) {
  auto ii = symbols.insert(str);
  return &*ii.first;
}

unsigned Galois::Runtime::StatCollector::getInstanceNum(const std::string& str) const {
  auto s = getSymbol(str);
  if (!s)
    return 0;
  auto ii = std::lower_bound(loopInstances.begin(), loopInstances.end(), s, [] (const StringPair<unsigned>& s1, const std::string* s2) { return s1.first < s2; } );
  if (ii == loopInstances.end() || s != ii->first)
    return 0;
  return ii->second;
}

void Galois::Runtime::StatCollector::addInstanceNum(const std::string& str) {
  auto s = getOrInsertSymbol(str);
  auto ii = std::lower_bound(loopInstances.begin(), loopInstances.end(), s, [] (const StringPair<unsigned>& s1, const std::string* s2) { return s1.first < s2; } );
  if (ii == loopInstances.end() || s != ii->first) {
    loopInstances.emplace_back(s, 0);
    std::sort(loopInstances.begin(), loopInstances.end(), [] (const StringPair<unsigned>& s1, const StringPair<unsigned>& s2) { return s1.first < s2.first; } );
  } else {
    ++ii->second;
  }
}

Galois::Runtime::StatCollector::RecordTy::RecordTy(size_t value) :mode(0), valueInt(value) {}

Galois::Runtime::StatCollector::RecordTy::RecordTy(double value) :mode(1), valueDouble(value) {}

Galois::Runtime::StatCollector::RecordTy::RecordTy(const std::string& value) :mode(2), valueStr(value) {}

Galois::Runtime::StatCollector::RecordTy::~RecordTy() {
  using string_type = std::string;
  if (mode == 2)
    valueStr.~string_type();
}

Galois::Runtime::StatCollector::RecordTy::RecordTy(const RecordTy& r) : mode(r.mode) {
  switch(mode) {
  case 0: valueInt    = r.valueInt;    break;
  case 1: valueDouble = r.valueDouble; break;
  case 2: new (&valueStr) std::string(r.valueStr);    break;
  }
}

void Galois::Runtime::StatCollector::RecordTy::print(std::ostream& out) const {
  switch(mode) {
  case 0: out << valueInt;    break;
  case 1: out << valueDouble; break;
  case 2: out << valueStr;    break;
  }
}

void Galois::Runtime::StatCollector::addToStat(const std::string& loop, const std::string& category, size_t value, unsigned TID, unsigned HostID) {
  MAKE_LOCK_GUARD(StatsLock);
  auto tpl = std::make_tuple(HostID, TID, getOrInsertSymbol(loop), getOrInsertSymbol(category), getInstanceNum(loop));
  auto iip = Stats.insert(std::make_pair(tpl, RecordTy(value)));
  if (iip.second == false) {
    assert(iip.first->second.mode == 0);
    iip.first->second.valueInt += value;
  }
}

void Galois::Runtime::StatCollector::addToStat(const std::string& loop, const std::string& category, double value, unsigned TID, unsigned HostID) {
  MAKE_LOCK_GUARD(StatsLock);
  auto tpl = std::make_tuple(HostID, TID, getOrInsertSymbol(loop), getOrInsertSymbol(category), getInstanceNum(loop));
  auto iip = Stats.insert(std::make_pair(tpl, RecordTy(value)));
  if (iip.second == false) {
    assert(iip.first->second.mode == 1);
    iip.first->second.valueDouble += value;
  }
}

void Galois::Runtime::StatCollector::addToStat(const std::string& loop, const std::string& category, const std::string& value, unsigned TID, unsigned HostID) {
  MAKE_LOCK_GUARD(StatsLock);
  auto tpl = std::make_tuple(HostID, TID, getOrInsertSymbol(loop), getOrInsertSymbol(category), getInstanceNum(loop));
  auto iip = Stats.insert(std::make_pair(tpl, RecordTy(value)));
  if (iip.second == false) {
    assert(iip.first->second.mode == 2);
    iip.first->second.valueStr = value;
  }
}

boost::uuids::uuid Galois::Runtime::StatCollector::UUID;
boost::uuids::uuid Galois::Runtime::StatCollector::getUUID(){
  if(UUID.is_nil()){
    boost::uuids::random_generator generator;
    UUID = generator();
    return UUID;
  }
  else {
    return UUID;
  }
}
//assumne called serially
void Galois::Runtime::StatCollector::printStatsForR(std::ostream& out, bool json) {
  if (json)
    out << "[\n";
  else
    out << "LOOP,INSTANCE,CATEGORY,THREAD,HOST,VAL\n";
  MAKE_LOCK_GUARD(StatsLock);
  for (auto& p : Stats) {
    if (json)
      out << "{\"UUID\" : " <<  getUUID() << ", \"LOOP\" : " << *std::get<2>(p.first) << " , \"INSTANCE\" : " << std::get<4>(p.first) << " , \"CATEGORY\" : " << *std::get<3>(p.first) << " , \"HOST\" : " << std::get<0>(p.first) << " , \"THREAD\" : " << std::get<1>(p.first) << " , \"VALUE\" : ";
    else
      out <<getUUID() <<"," << *std::get<2>(p.first) << "," << std::get<4>(p.first) << " , " << *std::get<3>(p.first) << "," << std::get<0>(p.first) << "," << std::get<1>(p.first) << ",";
    p.second.print(out);
    out << (json ? "}\n" : "\n");
  }
  if (json)
    out << "]\n";
}

void Galois::Runtime::StatCollector::printStats(void) {
  if (m_outfile == "") {
    printStats(std::cout);

  } else {
    std::ofstream outf(m_outfile);
    if (outf.good()) {
      printStatsForR(outf, false);
    } else {
      gWarn("Could not open stats file for writing, file provided:", m_outfile);
      printStats(std::cerr);
    }
  }
}

//Assume called serially
//still assumes int values
//This ignores HostID.  Thus this is not safe for the network version
void Galois::Runtime::StatCollector::printStats(std::ostream& out) {
  std::map<std::tuple<const std::string*, unsigned, const std::string*>, std::vector<size_t> > LKs;
  unsigned maxThreadID = 0;
  //Find all loops and keys
  MAKE_LOCK_GUARD(StatsLock);
  for (auto& p : Stats) {
    auto& v = LKs[std::make_tuple(std::get<2>(p.first), std::get<4>(p.first), std::get<3>(p.first))];
    auto tid = std::get<1>(p.first);
    maxThreadID = std::max(maxThreadID, tid);
    if (v.size() <= tid)
      v.resize(tid+1);
    v[tid] += p.second.valueInt;
  }
  //print header
  out << "STATTYPE,LOOP,INSTANCE,CATEGORY,n,sum";
  for (unsigned x = 0; x <= maxThreadID; ++x)
    out << ",T" << x;
  out << "\n";
  //print all values
  for (auto ii = LKs.begin(), ee = LKs.end(); ii != ee; ++ii) {
    auto& Values = ii->second;
    out << "STAT,"
        << std::get<0>(ii->first)->c_str() << ","
        << std::get<1>(ii->first) << ","
        << std::get<2>(ii->first)->c_str() << ","
        << maxThreadID + 1 <<  ","
        << std::accumulate(Values.begin(), Values.end(), static_cast<unsigned long>(0));
    for (unsigned x = 0; x <= maxThreadID; ++x)
      out << "," <<  (x < Values.size() ? Values.at(x) : 0);
    out << "\n";
  }
}

void Galois::Runtime::StatCollector::beginLoopInstance(const std::string& str) {
  addInstanceNum(str);
}

void Galois::Runtime::reportLoopInstance(const char* loopname) {
  SC->beginLoopInstance(std::string(loopname ? loopname : "(NULL)"));
}

void Galois::Runtime::reportStat(const char* loopname, const char* category, unsigned long value, unsigned TID) {
  SC->addToStat(std::string(loopname ? loopname : "(NULL)"), 
		      std::string(category ? category : "(NULL)"),
		      value, TID, 0);
}
void Galois::Runtime::reportStat(const char* loopname, const char* category, const std::string& value, unsigned TID) {
  SC->addToStat(std::string(loopname ? loopname : "(NULL)"), 
		      std::string(category ? category : "(NULL)"),
		      value, TID, 0);
}

void Galois::Runtime::reportStat(const std::string& loopname, const std::string& category, unsigned long value, unsigned TID) {
  SC->addToStat(loopname, category, value, TID, 0);
}

void Galois::Runtime::reportStat(const std::string& loopname, const std::string& category, const std::string& value, unsigned TID) {
  SC->addToStat(loopname, category, value, TID, 0);
}

void Galois::Runtime::reportPageAlloc(const char* category) {
  for (unsigned x = 0; x < Galois::Runtime::activeThreads; ++x)
    reportStat("(NULL)", category, static_cast<unsigned long>(numPagePoolAllocForThread(x)), x);
}

void Galois::Runtime::reportNumaAlloc(const char* category) {
  int nodes = Substrate::getThreadPool().getMaxNumaNodes();
  for (int x = 0; x < nodes; ++x) {
    //auto rStat = Stats.getRemote(x);
    //std::lock_guard<Substrate::SimpleLock> lg(rStat->first);
    //      rStat->second.emplace_back(loop, category, numNumaAllocForNode(x));
  }
  //  SC->addNumaAllocToStat(std::string("(NULL)"), std::string(category ? category : "(NULL)"));
}
