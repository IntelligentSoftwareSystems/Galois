/** DistStatManager Implementation -*- C++ -*-
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
 * @author M. Amber Hassaan<ahassaan@ices.utexas.edu>
 */

#include "galois/runtime/DistStats.h"
#include "galois/runtime/Serialize.h"

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

DistStatManager::DistStatManager(const std::string& outfile): StatManager(outfile) {}


class galois::runtime::StatRecvHelper {

public:

    static void recvAtHost_0_hostTotalTy(uint32_t hostID, galois::gstl::Str region, galois::gstl::Str category
        , StatTotal::Type totalTy) {


      dsm()->addRecvdHostTotalTy(hostID, region, category, totalTy);
    }

    static void recvAtHost_0_int(uint32_t hostID, galois::gstl::Str region, galois::gstl::Str category
        , int64_t thrdTotal,  StatTotal::Type totalTy, const galois::gstl::Vector<int64_t> thrdVals) {

      dsm()->addRecvdStat(hostID, region, category, thrdTotal, totalTy, thrdVals);
    }

    static void recvAtHost_0_fp(uint32_t hostID, galois::gstl::Str region, galois::gstl::Str category
        , double thrdTotal,  StatTotal::Type totalTy, const galois::gstl::Vector<double> thrdVals) {

      dsm()->addRecvdStat(hostID, region, category, thrdTotal, totalTy, thrdVals);
    }

    static void recvAtHost_0_str(uint32_t hostID, galois::gstl::Str region, galois::gstl::Str category
        , galois::gstl::Str thrdTotal,  StatTotal::Type totalTy, const galois::gstl::Vector<galois::gstl::Str> thrdVals) {

      dsm()->addRecvdParam(hostID, region, category, thrdTotal, totalTy, thrdVals);
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

    if (!IS_HOST0) {
      for (auto i = hTotalMap.cbegin(), end_i = hTotalMap.cend(); i != end_i; ++i) {

        getSystemNetworkInterface().sendSimple(0, StatRecvHelper::recvAtHost_0_hostTotalTy
            , hTotalMap.region(i), hTotalMap.category(i), hTotalMap.stat(i).totalTy());
      }
    }

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
        getSystemNetworkInterface().sendSimple(0, StatRecvHelper::recvAtHost_0_int, ln, cat, thrdTotal, totalTy, thrdVals);
      }
    }

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
        getSystemNetworkInterface().sendSimple(0, StatRecvHelper::recvAtHost_0_fp, ln, cat, thrdTotal, totalTy, thrdVals);
      }
    }

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
        getSystemNetworkInterface().sendSimple(0, StatRecvHelper::recvAtHost_0_str, ln, cat, thrdTotal, totalTy, thrdVals);
      }
    }

}


void DistStatManager::combineAtHost_0(void) {
  // first host 0 reads stats from Base class 
  // then barrier
  // then other hosts send stats to host 0
  // another barrier

  if (getHostID() == 0) {
    combineAtHost_0_helper();
  }

  galois::runtime::getHostBarrier().wait();

  if (getHostID() != 0) {
    combineAtHost_0_helper();
  }

  getSystemNetworkInterface().flush();

  galois::runtime::getHostBarrier().wait();


}

bool DistStatManager::printingHostVals(void) {
  return galois::substrate::EnvCheck(DistStatManager::HSTAT_ENV_VAR);
}

StatTotal::Type DistStatManager::findHostTotalTy(const Str& region, const Str& category, const StatTotal::Type& thrdTotalTy) const {

  StatTotal::Type hostTotalTy = thrdTotalTy;

  auto& mrgMap = hostTotalTypes.mergedMap();

  auto i = mrgMap.findStat(region, category);
  if (i != mrgMap.cend()) { 
    hostTotalTy = mrgMap.stat(i).totalTy(); 
  } 

  return hostTotalTy;
}

void DistStatManager::addRecvdHostTotalTy(unsigned hostID, const Str& region, const Str& category, const StatTotal::Type& totalTy) {
  hostTotalTypes.addToStat(region, category, totalTy);
}

void DistStatManager::addRecvdStat(unsigned hostID, const Str& region, const Str& category, int64_t thrdTotal, const StatTotal::Type& thrdTotalTy, const DistStatManager::ThrdVals<int64_t>& thrdVals) {

  intDistStats.addToStat(region, category
      , std::make_tuple(hostID, thrdTotal, thrdTotalTy, thrdVals)
      ,  findHostTotalTy(region, category, thrdTotalTy));

}

void DistStatManager::addRecvdStat(unsigned hostID, const Str& region, const Str& category, double thrdTotal, const StatTotal::Type& thrdTotalTy, const DistStatManager::ThrdVals<double>& thrdVals) {

  fpDistStats.addToStat(region, category
      , std::make_tuple(hostID, thrdTotal, thrdTotalTy, thrdVals)
      ,  findHostTotalTy(region, category, thrdTotalTy));

}

void DistStatManager::addRecvdParam(unsigned hostID, const Str& region, const Str& category, const Str& thrdTotal, const StatTotal::Type& thrdTotalTy, const DistStatManager::ThrdVals<Str>& thrdVals) {

  strDistStats.addToStat(region, category
      , std::make_tuple(hostID, thrdTotal, thrdTotalTy, thrdVals)
      ,  findHostTotalTy(region, category, thrdTotalTy));

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

  if (getHostID() == 0) {
    printHeader(out);

    intDistStats.print(out);
    fpDistStats.print(out);
    strDistStats.print(out);
  }
  // all hosts must wait for host 0 to finish printing stats
  //galois::gDebug("[", galois::runtime::getSystemNetworkInterface().ID, "] entering barrier\n");
  galois::runtime::getHostBarrier().wait();
  //galois::gDebug("[", galois::runtime::getSystemNetworkInterface().ID, "] leaving barrier\n");
  // delete/reset the system network interface
  galois::runtime::resetSystemNetworkInterface();
}
