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

using namespace Galois::Runtime;

boost::uuids::uuid Galois::Runtime::getRandUUID(void) {
  static boost::uuids::uuid UUID (boost::uuids::random_generator());
  return UUID;
}


StatManager(const Str& outfile): m_outfile(outfile) {}

void StatManager::print(void) {
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

void StatManager::printStats(std::ostream& out) {
  printHeader(out);
  intStats.print(out);
  fpStats.print(out);
}

unsigned StatManager::maxThreads(void) const { 
  return Substrate::getThreadPool().maxThreads();
}

void StatManager::printHeader(std::ostream& out) {

  // out << "RUN_UUID" << SEP;
  out << "STAT_TYPE" << SEP << "LOOPNAME" << SEP << "CATEGORY" << SEP;
  out << "THREAD_SUM" << SEP << "THREAD_AVG" << SEP << "THREAD_MIN" << SEP << "THREAD_MAX";


  for (unsigned i = 0; i < StatManager::maxThreads(); ++i) {
    out << SEP << "T" << i;
  }
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


void Galois::Runtime::reportParam(const char* loopname, const char* category, const std::string& value) {
  std::abort();
}


void Galois::Runtime::reportParam(const gstl::String& loopname, const gstl::String& category, const gstl::String& value) {
  std::abort();
}

void Galois::Runtime::reportPageAlloc(const char* category) {
  Runtime::on_each_gen(
      [] (const unsigned tid, const unsigned numT) {
        reportStat("(NULL)", category, numPagePoolAllocForThread(tid)); 
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

