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

#include "Galois/Runtime/StatCollector.h"
#include "Galois/Runtime/Support.h"
// #include "Galois/Runtime/Network.h"
#include "Galois/Runtime/Substrate.h"
#include "Galois/Substrate/StaticInstance.h"

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

  //Print header only on HOST 0
  if(getHostID() == 0){
    if (json)
      out << "[\n";
    else
      out << "LOOP,INSTANCE,CATEGORY,HOST,THREAD,VAL\n";
  }
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

  //  auto& net = Galois::Runtime::getSystemNetworkInterface();
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

//Implementation

static Substrate::StaticInstance<Galois::Runtime::StatCollector> SM;

void Galois::Runtime::reportLoopInstance(const char* loopname) {
  SM.get()->beginLoopInstance(std::string(loopname ? loopname : "(NULL)"));
}

static void reportStatImpl(uint32_t HostID, const std::string loopname, const std::string category, unsigned long value, unsigned TID) {
  if (getHostID())
    getSystemNetworkInterface().sendSimple(0, reportStatImpl, loopname, category, value, TID);
  else 
    SM.get()->addToStat(loopname, category, value, TID, HostID);
}

static void reportStatImpl(uint32_t HostID, const std::string loopname, const std::string category, const std::string value, unsigned TID) {
  if (getHostID())
    getSystemNetworkInterface().sendSimple(0, reportStatImpl, loopname, category, value, TID);
  else 
    SM.get()->addToStat(loopname, category, value, TID, HostID);
}

// TODO: following 4 variants are duplicated from libruntime/src/Support.cpp
void Galois::Runtime::reportStat(const std::string& loopname, const std::string& category, unsigned long value, unsigned TID) {
  reportStatImpl(getHostID(), loopname, category, value, TID);
}

void Galois::Runtime::reportStat(const std::string& loopname, const std::string& category, const std::string &value, unsigned TID) {
  reportStatImpl(getHostID(), loopname, category, value, TID);
  //out << loopname <<  ","  << 0 << ","<< category << "," << getHostID() << "," << TID << "," << value<<"\n";
}

void Galois::Runtime::reportStat(const char* loopname, const char* category, unsigned long value, unsigned TID) {
  reportStatImpl(getHostID(),
                 std::string(loopname ? loopname : "(NULL)"), 
                 std::string(category ? category : "(NULL)"),
                 value, TID);
}

void Galois::Runtime::reportStat(const char* loopname, const char* category, const std::string &value, unsigned TID) {
  reportStatImpl(getHostID(),
                 std::string(loopname ? loopname : "(NULL)"), 
                 std::string(category ? category : "(NULL)"),
                 value, TID);
}

static std::ofstream& openIfNot_output(std::string fname) {
  static std::ofstream output_file;
  if(!output_file.is_open()){
    output_file.open(fname, std::ios_base::app);
  }
  assert(output_file.is_open());
  return output_file;
}

void Galois::Runtime::printStats() {
  //getSystemNetworkInterface().reportStats();
  Galois::Runtime::getHostBarrier().wait();
  //SM.get()->printDistStats(std::cout);
  //SM.get()->printStats(std::cout);
  SM.get()->printStatsForR(std::cout, false);
}

// TODO: duplicated in libruntime/src/Support.cpp. Use that
void Galois::Runtime::printStats(std::string fname) {
  //getSystemNetworkInterface().reportStats();
  Galois::Runtime::getHostBarrier().wait();
  //SM.get()->printDistStats(std::cout);
  //SM.get()->printStats(std::cout);
  if(fname == "")
    SM.get()->printStatsForR(std::cout, false);
  else{
    auto& out = openIfNot_output(fname);
    SM.get()->printStatsForR(out, false);
    out.close();
  }
  if (getHostID() == 0) {
    std::cerr << "STAT FILENAME : " << fname << "\n";
  }
  Galois::Runtime::getHostBarrier().wait();
}
