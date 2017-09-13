/** Statistics collection and management -*- C++ -*-
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
 * Copyright (C) 2017, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */

#include "Galois/Runtime/Statistics.h"
#include "Galois/Runtime/Executor_OnEach.h"

#include <iostream>
#include <fstream>

using namespace Galois::Runtime;

boost::uuids::uuid Galois::Runtime::getRandUUID(void) {
  static boost::uuids::uuid UUID = boost::uuids::random_generator()();
  return UUID;
}

constexpr const char* StatTotal::NAMES[];

using Galois::gstl::Str;

StatManager::StatManager(const std::string& outfile): m_outfile(outfile) {}

StatManager::~StatManager(void) {}

bool StatManager::printingThreadVals(void) {
  return Galois::Substrate::EnvCheck(StatManager::TVAL_EVN_VAR);
}

void StatManager::addToStat(const Str& region, const Str& category, int64_t val, const StatTotal::Type& type) {
  intStats.addToStat(region, category, val, type);
}

void StatManager::addToStat(const Str& region, const Str& category, double val, const StatTotal::Type& type) {
  fpStats.addToStat(region, category, val, type);
}

void StatManager::addToParam(const Str& region, const Str& category, const Str& val) {
  strStats.addToStat(region, category, val, StatTotal::SERIAL);
}

void StatManager::print(void) {
  if (m_outfile == "") {
    printStats(std::cout);

  } else {
    std::ofstream outf(m_outfile.c_str());
    if (outf.good()) {
      printStats(outf);
    } else {
      gWarn("Could not open stats file for writing, file provided:", m_outfile);
      printStats(std::cerr);
    }
  }
}

void StatManager::printStats(std::ostream& out) {
  printHeader(out);
  intStats.print(out);
  fpStats.print(out);
  strStats.print(out);
}

unsigned StatManager::maxThreads(void) { 
  return Galois::getActiveThreads();
}

void StatManager::printHeader(std::ostream& out) {

  // out << "RUN_UUID" << SEP;
  out << "STAT_TYPE" << SEP << "REGION" << SEP << "CATEGORY" << SEP;
  out << "TOTAL_TYPE" << SEP << "TOTAL";
  out << std::endl;


}

StatManager::int_iterator StatManager::intBegin(void) const { return intStats.cbegin(); }
StatManager::int_iterator StatManager::intEnd(void) const { return intStats.cend(); }

StatManager::fp_iterator StatManager::fpBegin(void) const { return fpStats.cbegin(); }
StatManager::fp_iterator StatManager::fpEnd(void) const { return fpStats.cend(); }


static Galois::Runtime::StatManager* SM;

void Galois::Runtime::internal::setSysStatManager(Galois::Runtime::StatManager* sm) {
  GALOIS_ASSERT(!(SM && sm), "StatManager.cpp: Double Initialization of SM");
  SM = sm;
}

StatManager* Galois::Runtime::internal::sysStatManager(void) {
  return SM;
}


void Galois::Runtime::reportPageAlloc(const char* category) {
  Runtime::on_each_gen(
      [category] (const unsigned tid, const unsigned numT) {
        reportStat_Tsum("(NULL)", category, numPagePoolAllocForThread(tid)); 
      }
      , std::make_tuple(Galois::no_stats()));
}

void Galois::Runtime::reportNumaAlloc(const char* category) {
  Galois::gWarn("reportNumaAlloc NOT IMPLEMENTED YET. TBD");
  int nodes = Substrate::getThreadPool().getMaxNumaNodes();
  for (int x = 0; x < nodes; ++x) {
    //auto rStat = Stats.getRemote(x);
    //std::lock_guard<Substrate::SimpleLock> lg(rStat->first);
    //      rStat->second.emplace_back(loop, category, numNumaAllocForNode(x));
  }
  //  SC->addNumaAllocToStat(std::string("(NULL)"), std::string(category ? category : "(NULL)"));
}

