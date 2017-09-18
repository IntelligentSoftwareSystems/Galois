/** DistStatManager: Distributed Statistics Management -*- C++ -*-
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

#ifndef GALOIS_RUNTIME_DIST_STATS_H
#define GALOIS_RUNTIME_DIST_STATS_H

#include "Galois/gstl.h"
#include "Galois/Runtime/Statistics.h"
#include "Galois/Runtime/Network.h"

#include <string>

namespace Galois {
namespace Runtime {

class StatRecvHelper;

class DistStatManager: public Galois::Runtime::StatManager {

  friend class Galois::Runtime::StatRecvHelper;

  using Base = Galois::Runtime::StatManager;

  using Str = Galois::gstl::Str;

  using Base::SEP;

  static constexpr const char* const HSTAT_SEP = Base::TSTAT_SEP;
  static constexpr const char* const HSTAT_NAME = "HostValues";
  static constexpr const char* const HSTAT_ENV_VAR = "PRINT_PER_HOST_STATS";

  static bool printingHostVals(void);

  template <typename _UNUSED=void>
  struct HostTotalTypesImpl { 

    struct DummyStat {

      StatTotal::Type m_totalTy;

      explicit DummyStat(StatTotal::Type total): m_totalTy(total) {}

      template <typename _U>
      void add(const _U&) const {}

      const StatTotal::Type& totalTy(void) const { return m_totalTy; }
    };

    using TMap = hidden::BasicStatMap<DummyStat>;

    bool merged = false;
    Substrate::PerThreadStorage<TMap> perThrdMap;

    void addToStat(const Str& region, const Str& category, const StatTotal::Type& hostTotal) {
      perThrdMap.getLocal()->addToStat(region, category, 0, hostTotal);
    }

    void mergeStats(void) {
      if (merged) { return; }

      GALOIS_ASSERT(perThrdMap.getLocal() == perThrdMap.getRemote(0), "Must call from Thread 0");

      auto* t0Map = perThrdMap.getRemote(0);

      for (unsigned t = 1; t < perThrdMap.size(); ++t) {

        const auto* manager = perThrdMap.getRemote(t);

        for (auto i = manager->cbegin(), end_i = manager->cend(); i != end_i; ++i) {
          t0Map->addToStat(manager->region(i), manager->category(i), 0, manager->stat(i).totalTy());
        }
      }

      merged = true;
    }

    const TMap& mergedMap(void) const { 
      assert(merged && "Must merge first");
      return  *perThrdMap.getRemote(0);
    }
  };

  using HostTotalTypes = HostTotalTypesImpl<>;

  template <typename T>
  using ThrdVals = Galois::gstl::Vector<T>;

  template <typename T>
  using HostStatVal = std::tuple<unsigned, T, StatTotal::Type, const ThrdVals<T>&>;

  template <typename T>
  struct HostStat: public hidden::VecStat<T> {

    using Base = hidden::VecStat<T>;
    using ThrdStats = hidden::VecStat<T>;
    using PerHostThrdStats = Galois::gstl::Map<unsigned, ThrdStats>;

    PerHostThrdStats perHostThrdStats;

    explicit HostStat(const StatTotal::Type& hostTotal): Base(hostTotal) {}

    void add(const HostStatVal<T>& val) {

      const auto& hostID = std::get<0>(val);
      const auto& thrdTotal  = std::get<1>(val);
      const auto& thrdTotalTy  = std::get<2>(val);
      const auto& thrdVals  = std::get<3>(val);

      Base::add(thrdTotal);

      auto p = perHostThrdStats.emplace(hostID, ThrdStats(thrdTotalTy));
      auto& tstat = p.first->second;

      for (const auto& i: thrdVals) {
        tstat.add(i);
      }

    }

    void printHostVals(std::ostream& out, const Str& region, const Str& category) const {

      out << StatManager::statKind<T>() << SEP << Galois::Runtime::getHostID() << SEP;

      out << region << SEP << category << SEP;

      out << HSTAT_NAME << SEP;

      const char* sep = "";

      for (const auto& v: Base::values()) {
        out << sep << v;
        sep = HSTAT_SEP;
      }

      out << std::endl;
    }

    void printThreadVals(std::ostream& out, const Str& region, const Str& category) const {
      for (const auto& p: perHostThrdStats) {

        out << StatManager::statKind<T>() << SEP << p.first << SEP;
        out << region << SEP << category << SEP;

        out << StatTotal::str(p.second.totalTy()) << SEP << p.second.total();

        out << std::endl;

        out << StatManager::statKind<T>() << SEP << p.first << SEP;
        out << region << SEP << category << SEP;

        out << StatManager::TSTAT_NAME << SEP;

        const char* sep = "";
        for (const auto& v: p.second.values()) {
          out << sep << v;
          sep = StatManager::TSTAT_SEP;
        }

        out << std::endl;
      }
    }

  };

  template <typename T>
  struct DistStatCombiner: public hidden::BasicStatMap<HostStat<T> > {

    using Base = hidden::BasicStatMap<HostStat<T> >;

    static constexpr const char* htotalName(const StatTotal::Type& type) {
      switch(type) {
        case StatTotal::SERIAL: return "HOST_0";
        case StatTotal::TSUM: return "HSUM";
        case StatTotal::TAVG: return "HAVG";
        case StatTotal::TMIN: return "HMIN";
        case StatTotal::TMAX: return "HMAX";
        default: std::abort(); return nullptr;
      }
    }

    void print(std::ostream& out) const {

      for (auto i = Base::cbegin(), end_i = Base::cend(); i != end_i; ++i) {

        out << StatManager::statKind<T>() << SEP << Galois::Runtime::getHostID() << SEP;

        out << Base::region(i) << SEP << Base::category(i) << SEP;

        const HostStat<T>& hs = Base::stat(i);

        out << htotalName(hs.totalTy()) << SEP << hs.total();
        out << std::endl;

        if (DistStatManager::printingHostVals()) {
          hs.printHostVals(out, Base::region(i), Base::category(i));
        }

        if (StatManager::printingThreadVals()) {
          hs.printThreadVals(out, Base::region(i), Base::category(i));
        }
      }
    }

  };
  
  DistStatCombiner<int64_t> intDistStats;
  DistStatCombiner<double> fpDistStats;
  DistStatCombiner<Str> strDistStats;
  HostTotalTypes hostTotalTypes;

protected:
  void mergeStats(void);

  void printHeader(std::ostream& out) const;

  virtual void printStats(std::ostream& out);

public:

  DistStatManager(const std::string& outfile="");

  template <typename T>
  void addToStat(const Str& region, const Str& category, const T& val, const StatTotal::Type& threadTotal, const StatTotal::Type& hostTotal) {

    Base::addToStat(region, category, val, threadTotal);
    hostTotalTypes.addToStat(region, category, hostTotal);
  }

private:

  void combineAtHost_0_helper(void);

  void combineAtHost_0(void);

  StatTotal::Type findHostTotalTy(const Str& region, const Str& category, const StatTotal::Type& thrdTotalTy) const; 

  void addRecvdHostTotalTy(unsigned hostID, const Str& region, const Str& category, const StatTotal::Type& totalTy); 

  void addRecvdStat(unsigned hostID, const Str& region, const Str& category, int64_t thrdTotal, const StatTotal::Type& thrdTotalTy, const ThrdVals<int64_t>& thrdVals); 

  void addRecvdStat(unsigned hostID, const Str& region, const Str& category, double thrdTotal, const StatTotal::Type& thrdTotalTy, const ThrdVals<double>& thrdVals); 

  void addRecvdParam(unsigned hostID, const Str& region, const Str& category, const Str& thrdTotal, const StatTotal::Type& thrdTotalTy, const ThrdVals<Str>& thrdVals); 

};

namespace internal {
  DistStatManager* distSysStatManager(void);
}


template <typename S1, typename S2, typename T>
inline void reportDistStat(const S1& region, const S2& category, const T& value, const StatTotal::Type& threadTotal, const StatTotal::Type& hostTotal) {

  internal::distSysStatManager()->addToStat(
      gstl::makeStr(region), gstl::makeStr(category), value, threadTotal, hostTotal);
}


} // end namespace Runtime
} // end namespace Galois


#endif// GALOIS_RUNTIME_DIST_STATS_H
