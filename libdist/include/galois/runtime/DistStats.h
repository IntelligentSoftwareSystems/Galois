/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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
 * @file DistStats.h
 *
 * Contains declaration of DistStatManager, which reports runtime statistics of
 * a distributed application in Galois.
 */

#ifndef GALOIS_RUNTIME_DIST_STATS_H
#define GALOIS_RUNTIME_DIST_STATS_H

//! Turn on if you want more distributed stats to be printed
#ifndef MORE_DIST_STATS
#define MORE_DIST_STATS 0
#endif
//! Turn on if you want more communication statistics to be printed
#ifndef GALOIS_COMM_STATS
#define GALOIS_COMM_STATS 0
#endif
//! Turn on if you want per-bulk-synchronous parallel timers to be printed
//! (otherwise all rounds are under 1 timer)
#ifndef GALOIS_PER_ROUND_STATS
#define GALOIS_PER_ROUND_STATS 0
#endif

#include "galois/runtime/Statistics.h"
#include "galois/runtime/Network.h"

#include <string>

namespace galois {
namespace runtime {

/**
 * Helper class for the DistStatManager that aids in receiving statistics
 */
class StatRecvHelper;

/**
 * Class responsible for tracking all statistics of a running distributed
 * Galois program and reporting them at the end of program execution.
 */
class DistStatManager : public galois::runtime::StatManager {
  //! Friend class that helps with receiving stats
  friend class galois::runtime::StatRecvHelper;
  using Base = galois::runtime::StatManager;
  using Str  = galois::gstl::Str;
  using Base::SEP;

  static constexpr const char* const HSTAT_SEP     = Base::TSTAT_SEP;
  static constexpr const char* const HSTAT_NAME    = "HostValues";
  static constexpr const char* const HSTAT_ENV_VAR = "PRINT_PER_HOST_STATS";

  static bool printingHostVals(void);

  template <typename _UNUSED = void>
  struct HostTotalTypesImpl {
    struct DummyStat {
      StatTotal::Type m_totalTy;

      explicit DummyStat(StatTotal::Type total) : m_totalTy(total) {}

      template <typename _U>
      void add(const _U&) const {}

      const StatTotal::Type& totalTy(void) const { return m_totalTy; }
    };

    using TMap = internal::BasicStatMap<DummyStat>;

    bool merged = false;
    substrate::PerThreadStorage<TMap> perThrdMap;

    void addToStat(const Str& region, const Str& category,
                   const StatTotal::Type& hTotalTy) {
      perThrdMap.getLocal()->addToStat(region, category, 0, hTotalTy);
    }

    void mergeStats(void) {
      if (merged) {
        return;
      }
      GALOIS_ASSERT(perThrdMap.getLocal() == perThrdMap.getRemote(0),
                    "Must call from Thread 0");

      auto* t0Map = perThrdMap.getRemote(0);

      for (unsigned t = 1; t < perThrdMap.size(); ++t) {
        const auto* manager = perThrdMap.getRemote(t);

        for (auto i = manager->cbegin(), end_i = manager->cend(); i != end_i;
             ++i) {
          t0Map->addToStat(manager->region(i), manager->category(i), 0,
                           manager->stat(i).totalTy());
        }
      }

      merged = true;
    }

    const TMap& mergedMap(void) const {
      assert(merged && "Must merge first");
      return *perThrdMap.getRemote(0);
    }
  };

  using HostTotalTypes = HostTotalTypesImpl<>;

  template <typename T>
  using ThrdVals = galois::gstl::Vector<T>;

  template <typename T>
  using HostStatVal =
      std::tuple<unsigned, T, StatTotal::Type, const ThrdVals<T>&>;

  template <typename T>
  struct HostStat : public internal::VecStat<T> {
    using Base             = internal::VecStat<T>;
    using ThrdStats        = internal::VecStat<T>;
    using PerHostThrdStats = galois::gstl::Map<unsigned, ThrdStats>;

    PerHostThrdStats perHostThrdStats;

    explicit HostStat(const StatTotal::Type& hTotalTy) : Base(hTotalTy) {}

    void add(const HostStatVal<T>& val) {
      const auto& hostID      = std::get<0>(val);
      const auto& thrdTotal   = std::get<1>(val);
      const auto& thrdTotalTy = std::get<2>(val);
      const auto& thrdVals    = std::get<3>(val);

      Base::add(thrdTotal);

      auto p      = perHostThrdStats.emplace(hostID, ThrdStats(thrdTotalTy));
      auto& tstat = p.first->second;

      for (const auto& i : thrdVals) {
        tstat.add(i);
      }
    }

    void printHostVals(std::ostream& out, const Str& region,
                       const Str& category) const {
      out << StatManager::statKind<T>() << SEP << galois::runtime::getHostID()
          << SEP;
      out << region << SEP << category << SEP;
      out << HSTAT_NAME << SEP;

      const char* sep = "";

      for (const auto& v : Base::values()) {
        out << sep << v;
        sep = HSTAT_SEP;
      }

      out << std::endl;
    }

    void printThreadVals(std::ostream& out, const Str& region,
                         const Str& category) const {
      for (const auto& p : perHostThrdStats) {
        out << StatManager::statKind<T>() << SEP << p.first << SEP;
        out << region << SEP << category << SEP;
        out << StatTotal::str(p.second.totalTy()) << SEP << p.second.total();
        out << std::endl;

        out << StatManager::statKind<T>() << SEP << p.first << SEP;
        out << region << SEP << category << SEP;
        out << StatManager::TSTAT_NAME << SEP;

        const char* sep = "";
        for (const auto& v : p.second.values()) {
          out << sep << v;
          sep = StatManager::TSTAT_SEP;
        }

        out << std::endl;
      }
    }
  };

  template <typename T>
  struct DistStatCombiner : public internal::BasicStatMap<HostStat<T>> {
    using Base = internal::BasicStatMap<HostStat<T>>;

#if __GNUC__ < 5
    static const char* htotalName(const StatTotal::Type& type){
#else
    static constexpr const char* htotalName(const StatTotal::Type& type) {
#endif
        switch (type) {
          case StatTotal::SINGLE : return "HOST_0";
  case StatTotal::TSUM:
    return "HSUM";
  case StatTotal::TAVG:
    return "HAVG";
  case StatTotal::TMIN:
    return "HMIN";
  case StatTotal::TMAX:
    return "HMAX";
  default:
    std::abort();
    return nullptr;
  }
}

    void print(std::ostream& out) const {
  for (auto i = Base::cbegin(), end_i = Base::cend(); i != end_i; ++i) {
    out << StatManager::statKind<T>() << SEP << galois::runtime::getHostID()
        << SEP;
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
}; // namespace runtime

DistStatCombiner<int64_t> intDistStats;
DistStatCombiner<double> fpDistStats;
DistStatCombiner<Str> strDistStats;
HostTotalTypes hostTotalTypes;

protected:
/**
 * Merge all stats from each individual thread as well as each individual
 * host as prescribed the the reduction (Total) type specified for each
 * statistic.
 */
void mergeStats(void);

/**
 * Print the header of the stats file output.
 *
 * @param out File to print header out to
 */
void printHeader(std::ostream& out) const;

/**
 * Merge all stats. Host 0 will then print out all collected stats.
 */
virtual void printStats(std::ostream& out);

public:
//! Dist stat manager constructor
DistStatManager(const std::string& outfile = "");
~DistStatManager();

/**
 * Adds a statistic to the statistics manager.
 *
 * @param region Region name to give statistic
 * @param category Category of statistic
 * @param val Value of the statistic
 * @param thrdTotalTy The type of reduction used to combine thread statistics
 * of the same kind
 * @param hTotalTy The type of reduction used to combine host statistics
 * of the same kind
 */
template <typename T>
void addToStat(const Str& region, const Str& category, const T& val,
               const StatTotal::Type& thrdTotalTy,
               const StatTotal::Type& hTotalTy) {
  Base::addToStat(region, category, val, thrdTotalTy);
  hostTotalTypes.addToStat(region, category, hTotalTy);
}

private:
void combineAtHost_0_helper(void);
void combineAtHost_0_helper2(void);
void receiveAtHost_0_helper(void);
void receiveAtHost_0_helper2(void);
void combineAtHost_0(void);
StatTotal::Type findHostTotalTy(const Str& region, const Str& category,
                                const StatTotal::Type& thrdTotalTy) const;
void addRecvdHostTotalTy(const Str& region, const Str& category,
                         const StatTotal::Type& totalTy);
void addRecvdStat(unsigned hostID, const Str& region, const Str& category,
                  int64_t thrdTotal, const StatTotal::Type& thrdTotalTy,
                  const ThrdVals<int64_t>& thrdVals);
void addRecvdStat(unsigned hostID, const Str& region, const Str& category,
                  double thrdTotal, const StatTotal::Type& thrdTotalTy,
                  const ThrdVals<double>& thrdVals);
void addRecvdParam(unsigned hostID, const Str& region, const Str& category,
                   const Str& thrdTotal, const StatTotal::Type& thrdTotalTy,
                   const ThrdVals<Str>& thrdVals);
}; // namespace galois

namespace internal {
/**
 * Gets a pointer to the distributed stat manager.
 *
 * @returns Pointer to distributed statistics manager
 */
DistStatManager* distSysStatManager(void);
} // namespace internal

/**
 * Adds a statistic to the statistics manager. Calls addToStat in
 * DistStatManager.
 *
 * @param region Region name to give statistic
 * @param category Category of statistic
 * @param value Value of the statistic
 * @param thrdTotalTy The type of reduction used to combine thread statistics
 * of the same kind
 * @param hTotalTy The type of reduction used to combine host statistics
 * of the same kind
 */
template <typename S1, typename S2, typename T>
inline void reportDistStat(const S1& region, const S2& category, const T& value,
                           const StatTotal::Type& thrdTotalTy,
                           const StatTotal::Type& hTotalTy) {
  internal::distSysStatManager()->addToStat(gstl::makeStr(region),
                                            gstl::makeStr(category), value,
                                            thrdTotalTy, hTotalTy);
}

} // end namespace runtime
} // end namespace galois

#endif // GALOIS_RUNTIME_DIST_STATS_H
