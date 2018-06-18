/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "galois/runtime/Statistics.h"
#include "galois/runtime/Executor_OnEach.h"

#include <iostream>
#include <fstream>

using namespace galois::runtime;

boost::uuids::uuid galois::runtime::getRandUUID(void) {
  static boost::uuids::uuid UUID = boost::uuids::random_generator()();
  return UUID;
}

using galois::gstl::Str;

StatManager::StatManager(const std::string& outfile) : m_outfile(outfile) {}

StatManager::~StatManager(void) {}

void StatManager::setStatFile(const std::string& outfile) {
  m_outfile = outfile;
}

void galois::runtime::setStatFile(const std::string& f) {
  internal::sysStatManager()->setStatFile(f);
}

bool StatManager::printingThreadVals(void) {
  return galois::substrate::EnvCheck(StatManager::TSTAT_ENV_VAR);
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
  mergeStats();
  printHeader(out);
  intStats.print(out);
  fpStats.print(out);
  strStats.print(out);
}

void StatManager::printHeader(std::ostream& out) const {

  // out << "RUN_UUID" << SEP;
  out << "STAT_TYPE" << SEP << "REGION" << SEP << "CATEGORY" << SEP;
  out << "TOTAL_TYPE" << SEP << "TOTAL";
  out << std::endl;
}

StatManager::int_iterator StatManager::intBegin(void) const {
  return intStats.cbegin();
}
StatManager::int_iterator StatManager::intEnd(void) const {
  return intStats.cend();
}

StatManager::fp_iterator StatManager::fpBegin(void) const {
  return fpStats.cbegin();
}
StatManager::fp_iterator StatManager::fpEnd(void) const {
  return fpStats.cend();
}

StatManager::str_iterator StatManager::paramBegin(void) const {
  return strStats.cbegin();
}
StatManager::str_iterator StatManager::paramEnd(void) const {
  return strStats.cend();
}

static galois::runtime::StatManager* SM;

void galois::runtime::internal::setSysStatManager(
    galois::runtime::StatManager* sm) {
  GALOIS_ASSERT(!(SM && sm), "StatManager.cpp: Double Initialization of SM");
  SM = sm;
}

StatManager* galois::runtime::internal::sysStatManager(void) { return SM; }

void galois::runtime::reportPageAlloc(const char* category) {
  galois::runtime::on_each_gen(
      [category](const unsigned tid, const unsigned numT) {
        reportStat_Tsum("PageAlloc", category, numPagePoolAllocForThread(tid));
      },
      std::make_tuple());
}

void galois::runtime::reportNumaAlloc(const char* category) {
  galois::gWarn("reportNumaAlloc NOT IMPLEMENTED YET. TBD");
  int nodes = substrate::getThreadPool().getMaxNumaNodes();
  for (int x = 0; x < nodes; ++x) {
    // auto rStat = Stats.getRemote(x);
    // std::lock_guard<substrate::SimpleLock> lg(rStat->first);
    //      rStat->second.emplace_back(loop, category, numNumaAllocForNode(x));
  }
  //  SC->addNumaAllocToStat(std::string("(NULL)"), std::string(category ?
  //  category : "(NULL)"));
}
