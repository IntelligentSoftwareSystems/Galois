/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
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

/**
 * @file DistStats.cpp
 *
 * Contains implementations for DistStats.h.
 */
#include "galois/runtime/DistStats.h"
#include "galois/runtime/Serialize.h"
#include "galois/DTerminationDetector.h"

using namespace galois::runtime;

DistStatManager* internal::distSysStatManager(void) {
  galois::runtime::StatManager* sm = internal::sysStatManager();

  assert(sm && "StatManager not initialized");

  DistStatManager* d = dynamic_cast<DistStatManager*>(sm);

  assert(d && "dynamic_cast<DistStatManager*> failed");

  return d;
}

inline static DistStatManager* dsm(void) {
  return internal::distSysStatManager();
}

DistStatManager::DistStatManager(const std::string& outfile)
    : StatManager(outfile) {}
DistStatManager::~DistStatManager() {
  galois::runtime::internal::destroySystemNetworkInterface();
}

class galois::runtime::StatRecvHelper {

public:
  static void recvAtHost_0_hostTotalTy(galois::gstl::Str region,
                                       galois::gstl::Str category,
                                       StatTotal::Type totalTy) {

    dsm()->addRecvdHostTotalTy(region, category, totalTy);
  }

  static void recvAtHost_0_int(uint32_t hostID, galois::gstl::Str region,
                               galois::gstl::Str category, int64_t thrdTotal,
                               StatTotal::Type totalTy,
                               const galois::gstl::Vector<int64_t> thrdVals) {

    dsm()->addRecvdStat(hostID, region, category, thrdTotal, totalTy, thrdVals);
  }

  static void recvAtHost_0_fp(uint32_t hostID, galois::gstl::Str region,
                              galois::gstl::Str category, double thrdTotal,
                              StatTotal::Type totalTy,
                              const galois::gstl::Vector<double> thrdVals) {

    dsm()->addRecvdStat(hostID, region, category, thrdTotal, totalTy, thrdVals);
  }

  static void
  recvAtHost_0_str(uint32_t hostID, galois::gstl::Str region,
                   galois::gstl::Str category, galois::gstl::Str thrdTotal,
                   StatTotal::Type totalTy,
                   const galois::gstl::Vector<galois::gstl::Str> thrdVals) {

    dsm()->addRecvdParam(hostID, region, category, thrdTotal, totalTy,
                         thrdVals);
  }
};

void DistStatManager::mergeStats(void) {
  Base::mergeStats();
  hostTotalTypes.mergeStats();
  combineAtHost_0();
}

void DistStatManager::combineAtHost_0_helper(void) {
  const bool IS_HOST0 = getHostID() == 0;

  const auto& hTotalMap = hostTotalTypes.mergedMap();

  size_t syncTypePhase = 0;
  if (!IS_HOST0) {
    for (auto i = hTotalMap.cbegin(), end_i = hTotalMap.cend(); i != end_i;
         ++i) {
      SendBuffer b;
      gSerialize(b, hTotalMap.region(i),
          hTotalMap.category(i), hTotalMap.stat(i).totalTy());
      getSystemNetworkInterface().sendTagged(0, galois::runtime::evilPhase,
          b, syncTypePhase);
    }
  }

  ++syncTypePhase;
  for (auto i = Base::intBegin(), end_i = Base::intEnd(); i != end_i; ++i) {
    Str ln;
    Str cat;
    int64_t thrdTotal;
    StatTotal::Type totalTy;
    galois::gstl::Vector<int64_t> thrdVals;

    Base::readIntStat(i, ln, cat, thrdTotal, totalTy, thrdVals);

    if (IS_HOST0) {
      addRecvdStat(0, ln, cat, thrdTotal, totalTy, thrdVals);

    } else {
      SendBuffer b;
      gSerialize(b, ln, cat, thrdTotal, totalTy, thrdVals);
      getSystemNetworkInterface().sendTagged(0, galois::runtime::evilPhase,
          b, syncTypePhase);
    }
  }
}

void DistStatManager::combineAtHost_0_helper2(void) {
  const bool IS_HOST0 = getHostID() == 0;

  size_t syncTypePhase = 0;
  for (auto i = Base::fpBegin(), end_i = Base::fpEnd(); i != end_i; ++i) {
    Str ln;
    Str cat;
    double thrdTotal;
    StatTotal::Type totalTy;
    galois::gstl::Vector<double> thrdVals;

    Base::readFPstat(i, ln, cat, thrdTotal, totalTy, thrdVals);

    if (IS_HOST0) {
      addRecvdStat(0, ln, cat, thrdTotal, totalTy, thrdVals);

    } else {
      SendBuffer b;
      gSerialize(b, ln, cat, thrdTotal, totalTy, thrdVals);
      getSystemNetworkInterface().sendTagged(0, galois::runtime::evilPhase,
          b, syncTypePhase);
    }
  }

  ++syncTypePhase;
  for (auto i = Base::paramBegin(), end_i = Base::paramEnd(); i != end_i; ++i) {
    Str ln;
    Str cat;
    Str thrdTotal;
    StatTotal::Type totalTy;
    galois::gstl::Vector<Str> thrdVals;

    Base::readParam(i, ln, cat, thrdTotal, totalTy, thrdVals);

    if (IS_HOST0) {
      addRecvdParam(0, ln, cat, thrdTotal, totalTy, thrdVals);

    } else {
      SendBuffer b;
      gSerialize(b, ln, cat, thrdTotal, totalTy, thrdVals);
      getSystemNetworkInterface().sendTagged(0, galois::runtime::evilPhase,
          b, syncTypePhase);
    }
  }
}

void DistStatManager::receiveAtHost_0_helper(void) {
  size_t syncTypePhase = 0;
  {
    decltype(getSystemNetworkInterface().recieveTagged(galois::runtime::evilPhase, nullptr, syncTypePhase)) p;
    do {
      p = getSystemNetworkInterface().recieveTagged(galois::runtime::evilPhase, nullptr, syncTypePhase);

      if (p) {
        RecvBuffer& b = p->second;

        galois::gstl::Str region;
        galois::gstl::Str category;
        StatTotal::Type totalTy;
        gDeserialize(b, region, category, totalTy);

        StatRecvHelper::recvAtHost_0_hostTotalTy(region, category, totalTy);
      }
    } while (p);
  }

  ++syncTypePhase;
  {
    decltype(getSystemNetworkInterface().recieveTagged(galois::runtime::evilPhase, nullptr, syncTypePhase)) p;
    do {
      p = getSystemNetworkInterface().recieveTagged(galois::runtime::evilPhase, nullptr, syncTypePhase);

      if (p) {
        uint32_t hostID = p->first;
        RecvBuffer& b = p->second;

        Str ln;
        Str cat;
        int64_t thrdTotal;
        StatTotal::Type totalTy;
        galois::gstl::Vector<int64_t> thrdVals;
        gDeserialize(b, ln, cat, thrdTotal, totalTy, thrdVals);

        StatRecvHelper::recvAtHost_0_int(hostID, ln, cat, thrdTotal, totalTy, thrdVals);
      }
    } while (p);
  }
}

void DistStatManager::receiveAtHost_0_helper2(void) {
  size_t syncTypePhase = 0;
  {
    decltype(getSystemNetworkInterface().recieveTagged(galois::runtime::evilPhase, nullptr, syncTypePhase)) p;
    do {
      p = getSystemNetworkInterface().recieveTagged(galois::runtime::evilPhase, nullptr, syncTypePhase);

      if (p) {
        uint32_t hostID = p->first;
        RecvBuffer& b = p->second;

        Str ln;
        Str cat;
        double thrdTotal;
        StatTotal::Type totalTy;
        galois::gstl::Vector<double> thrdVals;
        gDeserialize(b, ln, cat, thrdTotal, totalTy, thrdVals);

        StatRecvHelper::recvAtHost_0_fp(hostID, ln, cat, thrdTotal, totalTy, thrdVals);
      }
    } while (p);
  }

  ++syncTypePhase;
  {
    decltype(getSystemNetworkInterface().recieveTagged(galois::runtime::evilPhase, nullptr, syncTypePhase)) p;
    do {
      p = getSystemNetworkInterface().recieveTagged(galois::runtime::evilPhase, nullptr, syncTypePhase);

      if (p) {
        uint32_t hostID = p->first;
        RecvBuffer& b = p->second;

        Str ln;
        Str cat;
        Str thrdTotal;
        StatTotal::Type totalTy;
        galois::gstl::Vector<Str> thrdVals;
        gDeserialize(b, ln, cat, thrdTotal, totalTy, thrdVals);

        StatRecvHelper::recvAtHost_0_str(hostID, ln, cat, thrdTotal, totalTy, thrdVals);
      }
    } while (p);
  }
}

void DistStatManager::combineAtHost_0(void) {
  galois::DGTerminator<unsigned int> td;

  // host 0 reads stats from Base class
  // other hosts send stats to host 0
  combineAtHost_0_helper();
  getSystemNetworkInterface().flush();

  // barrier
  while (td.reduce()) {
    if (getHostID() == 0) {
      // receive from other hosts
      receiveAtHost_0_helper();
    }
  };
  // explicit barrier after logical barrier is required
  // as next async phase begins immediately
  getHostBarrier().wait();

  // host 0 reads stats from Base class
  // other hosts send stats to host 0
  combineAtHost_0_helper2();
  getSystemNetworkInterface().flush();

  // barrier
  while (td.reduce()) {
    if (getHostID() == 0) {
      // receive from other hosts
      receiveAtHost_0_helper2();
    }
  };
  // explicit barrier after logical barrier is required
  // as next async phase begins immediately
  getHostBarrier().wait();
}

bool DistStatManager::printingHostVals(void) {
  return galois::substrate::EnvCheck(DistStatManager::HSTAT_ENV_VAR);
}

StatTotal::Type
DistStatManager::findHostTotalTy(const Str& region, const Str& category,
                                 const StatTotal::Type& thrdTotalTy) const {

  StatTotal::Type hostTotalTy = thrdTotalTy;

  auto& mrgMap = hostTotalTypes.mergedMap();

  auto i = mrgMap.findStat(region, category);
  if (i != mrgMap.cend()) {
    hostTotalTy = mrgMap.stat(i).totalTy();
  }

  return hostTotalTy;
}

void DistStatManager::addRecvdHostTotalTy(const Str& region,
                                          const Str& category,
                                          const StatTotal::Type& totalTy) {
  hostTotalTypes.addToStat(region, category, totalTy);
}

void DistStatManager::addRecvdStat(
    unsigned hostID, const Str& region, const Str& category, int64_t thrdTotal,
    const StatTotal::Type& thrdTotalTy,
    const DistStatManager::ThrdVals<int64_t>& thrdVals) {

  intDistStats.addToStat(
      region, category,
      std::make_tuple(hostID, thrdTotal, thrdTotalTy, thrdVals),
      findHostTotalTy(region, category, thrdTotalTy));
}

void DistStatManager::addRecvdStat(
    unsigned hostID, const Str& region, const Str& category, double thrdTotal,
    const StatTotal::Type& thrdTotalTy,
    const DistStatManager::ThrdVals<double>& thrdVals) {

  fpDistStats.addToStat(
      region, category,
      std::make_tuple(hostID, thrdTotal, thrdTotalTy, thrdVals),
      findHostTotalTy(region, category, thrdTotalTy));
}

void DistStatManager::addRecvdParam(
    unsigned hostID, const Str& region, const Str& category,
    const Str& thrdTotal, const StatTotal::Type& thrdTotalTy,
    const DistStatManager::ThrdVals<Str>& thrdVals) {

  strDistStats.addToStat(
      region, category,
      std::make_tuple(hostID, thrdTotal, thrdTotalTy, thrdVals),
      findHostTotalTy(region, category, thrdTotalTy));
}

void DistStatManager::printHeader(std::ostream& out) const {
  out << "STAT_TYPE" << SEP;
  out << "HOST_ID" << SEP;
  out << "REGION" << SEP << "CATEGORY" << SEP;
  out << "TOTAL_TYPE" << SEP << "TOTAL";

  out << std::endl;
}

void DistStatManager::printStats(std::ostream& out) {
  mergeStats();

  galois::DGTerminator<unsigned int> td;
  if (getHostID() == 0) {
    printHeader(out);

    intDistStats.print(out);
    fpDistStats.print(out);
    strDistStats.print(out);
  }
  // all hosts must wait for host 0 to finish printing stats
  while (td.reduce()) {};
}
