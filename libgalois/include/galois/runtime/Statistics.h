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

#ifndef GALOIS_STAT_MANAGER_H
#define GALOIS_STAT_MANAGER_H

#include "galois/gstl.h"
#include "galois/gIO.h"
#include "galois/Threads.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/substrate/ThreadRWlock.h"
#include "galois/substrate/EnvCheck.h"

#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.

#include <limits>
#include <string>
#include <map>
#include <type_traits>
/**
 * TODO:
 * Print intra host stats with per-thread details and inter-host stats with
 per-host details
 * print to 2 files if supporting R format
 * dist implements an addToStat with host ID and manages inter-host stats and
 their combining

 */

namespace galois {
namespace runtime {

boost::uuids::uuid getRandUUID();

template <typename T>
class RunningMin {
  T m_min;

public:
  RunningMin(void) : m_min(std::numeric_limits<T>::max()) {}

  void add(const T& val) { m_min = std::min(m_min, val); }

  const T& min(void) const { return m_min; }
};

template <typename T>
class RunningMax {
  T m_max;

public:
  RunningMax(void) : m_max(std::numeric_limits<T>::min()) {}

  void add(const T& val) { m_max = std::max(m_max, val); }

  const T& max(void) const { return m_max; }
};

template <typename T>
class RunningSum {
  T m_sum;
  size_t m_count;

public:
  RunningSum(void) : m_sum(), m_count(0) {}

  void add(const T& val) {
    m_sum += val;
    ++m_count;
  }

  const T& sum(void) const { return m_sum; }

  const size_t& count(void) const { return m_count; }

  T avg() const { return m_sum / m_count; }
};

template <typename T>
class RunningVec {

  using Vec = gstl::Vector<T>;

  Vec m_vec;

public:
  void add(const T& val) { m_vec.push_back(val); }

  const Vec& values(void) const { return m_vec; }
};

template <typename T>
class NamedStat {

  using Str = galois::gstl::Str;

  Str m_name;

public:
  void setName(const Str& name) { m_name = name; }

  void setName(Str&& name) { m_name = std::move(name); }

  const Str& name(void) const { return m_name; }

  void add(const T& val) const {}
};

template <typename T, typename... Bases>
class AggregStat : public Bases... {

public:
  using with_min = AggregStat<T, RunningMin<T>, Bases...>;

  using with_max = AggregStat<T, RunningMax<T>, Bases...>;

  using with_sum = AggregStat<T, RunningSum<T>, Bases...>;

  using with_mem = AggregStat<T, RunningVec<T>, Bases...>;

  using with_name = AggregStat<T, NamedStat<T>, Bases...>;

  void add(const T& val) {
    using Expander = int[];

    (void)Expander{0, ((void)Bases::add(val), 0)...};
  }
};

namespace {
  static constexpr const char* StatTotalNames[] = {"SINGLE", "TMIN", "TMAX",
                                                   "TSUM", "TAVG"};
}

struct StatTotal {

  enum Type { SINGLE = 0, TMIN, TMAX, TSUM, TAVG };

  static const char* str(const Type& t) { return StatTotalNames[t]; }
};

namespace internal {

template <typename Stat_tp>
struct BasicStatMap {

  using Stat    = Stat_tp;
  using Str     = galois::gstl::Str;
  using StrSet  = galois::gstl::Set<Str>;
  using StatMap = galois::gstl::Map<std::tuple<const Str*, const Str*>, Stat>;
  using const_iterator = typename StatMap::const_iterator;

protected:
  StrSet symbols;
  StatMap statMap;

  const Str* getOrInsertSymbol(const Str& s) {
    auto p = symbols.insert(s);
    return &*(p.first);
  }

  const Str* getSymbol(const Str& s) const {
    auto i = symbols.find(s);

    if (i == symbols.cend()) {
      return nullptr;
    } else {
      return &(*i);
    }
  }

public:
  template <typename... Args>
  Stat& getOrInsertStat(const Str& region, const Str& category,
                        Args&&... args) {

    const Str* ln  = getOrInsertSymbol(region);
    const Str* cat = getOrInsertSymbol(category);

    auto tpl = std::make_tuple(ln, cat);

    auto p = statMap.emplace(tpl, Stat(std::forward<Args>(args)...));

    return p.first->second;
  }

  const_iterator findStat(const Str& region, const Str& category) const {

    const Str* ln  = getSymbol(region);
    const Str* cat = getSymbol(category);
    auto tpl       = std::make_tuple(ln, cat);

    auto i = statMap.find(tpl);

    return i;
  }

  const Stat& getStat(const Str& region, const Str& category) const {

    auto i = findStat(region, category);
    assert(i != statMap.end());
    return i->second;
  }

  template <typename T, typename... Args>
  void addToStat(const Str& region, const Str& category, const T& val,
                 Args&&... statArgs) {
    Stat& s =
        getOrInsertStat(region, category, std::forward<Args>(statArgs)...);
    s.add(val);
  }

  const_iterator cbegin(void) const { return statMap.cbegin(); }
  const_iterator cend(void) const { return statMap.cend(); }

  const Str& region(const const_iterator& i) const {
    return *(std::get<0>(i->first));
  }

  const Str& category(const const_iterator& i) const {
    return *(std::get<1>(i->first));
  }

  const Stat& stat(const const_iterator& i) const { return i->second; }
};

template <typename T>
using VecStat_with_MinMaxSum =
    typename AggregStat<T>::with_mem::with_min::with_max::with_sum;

template <typename T>
struct VecStat : public VecStat_with_MinMaxSum<T> {

  using Base = VecStat_with_MinMaxSum<T>;

  StatTotal::Type m_totalTy;

  explicit VecStat(const StatTotal::Type& type) : Base(), m_totalTy(type) {}

  const StatTotal::Type& totalTy(void) const { return m_totalTy; }

  T total(void) const {

    switch (m_totalTy) {

    case StatTotal::SINGLE:
      assert(Base::values().size() > 0);
      return Base::values()[0];

    case StatTotal::TMIN:
      return Base::min();

    case StatTotal::TMAX:
      return Base::max();

    case StatTotal::TSUM:
      return Base::sum();

    case StatTotal::TAVG:
      return Base::avg();

    default:
      GALOIS_DIE("Shouldn't reach this point");
    }
  }
};

template <>
struct VecStat<gstl::Str> : public AggregStat<gstl::Str>::with_mem {

  using Base = AggregStat<gstl::Str>::with_mem;

  StatTotal::Type m_totalTy;

  explicit VecStat(const StatTotal::Type& type) : Base(), m_totalTy(type) {}

  const StatTotal::Type& totalTy(void) const { return m_totalTy; }

  const gstl::Str& total(void) const {

    switch (m_totalTy) {

    case StatTotal::SINGLE:
      assert(Base::values().size() > 0);
      return Base::values()[0];

    default:
      GALOIS_DIE("Shouldn't reach this point. m_totalTy has unsupported value");
    }
  }
};

template <typename T>
using VecStatManager = BasicStatMap<VecStat<T>>;

template <typename T>
struct ScalarStat {
  T m_val;
  StatTotal::Type m_totalTy;

  explicit ScalarStat(const StatTotal::Type& type) : m_val(), m_totalTy(type) {}

  void add(const T& v) { m_val += v; }

  operator const T&(void)const { return m_val; }

  const StatTotal::Type& totalTy(void) const { return m_totalTy; }
};

template <typename T>
using ScalarStatManager = BasicStatMap<ScalarStat<T>>;

} // end namespace internal

#define STAT_MANAGER_IMPL 0 // 0 or 1 or 2

#if STAT_MANAGER_IMPL == 0

class StatManager {

public:
  using Str = galois::gstl::Str;

  static constexpr const char* const SEP           = ", ";
  static constexpr const char* const TSTAT_SEP     = "; ";
  static constexpr const char* const TSTAT_NAME    = "ThreadValues";
  static constexpr const char* const TSTAT_ENV_VAR = "PRINT_PER_THREAD_STATS";

  static bool printingThreadVals(void);

  template <typename T>
  static constexpr const char* statKind(void) {
    return std::is_same<T, Str>::value ? "PARAM" : "STAT";
  }

private:
  template <typename T>
  struct StatManagerImpl {

    using MergedStats    = internal::VecStatManager<T>;
    using const_iterator = typename MergedStats::const_iterator;
    using Stat           = typename MergedStats::Stat;

    substrate::PerThreadStorage<internal::ScalarStatManager<T>>
        perThreadManagers;
    MergedStats result;
    bool merged = false;

    void addToStat(const Str& region, const Str& category, const T& val,
                   const StatTotal::Type& type) {
      perThreadManagers.getLocal()->addToStat(region, category, val, type);
    }

    void mergeStats(void) {

      if (merged) {
        return;
      }

      for (unsigned t = 0; t < perThreadManagers.size(); ++t) {

        const auto* manager = perThreadManagers.getRemote(t);

        for (auto i = manager->cbegin(), end_i = manager->cend(); i != end_i;
             ++i) {
          result.addToStat(manager->region(i), manager->category(i),
                           T(manager->stat(i)), manager->stat(i).totalTy());
        }
      }

      merged = true;
    }

    const_iterator cbegin(void) const { return result.cbegin(); }
    const_iterator cend(void) const { return result.cend(); }

    const Str& region(const const_iterator& i) const {
      return result.region(i);
    }

    const Str& category(const const_iterator& i) const {
      return result.category(i);
    }

    const Stat& stat(const const_iterator& i) const { return result.stat(i); }

    template <typename S, typename V>
    void readStat(const const_iterator& i, S& region, S& category, T& total,
                  StatTotal::Type& type, V& thrdVals) const {
      region   = this->region(i);
      category = this->category(i);

      total = this->stat(i).total();
      type  = this->stat(i).totalTy();

      thrdVals.clear();
      thrdVals = this->stat(i).values();
    }

    void print(std::ostream& out) const {

      for (auto i = cbegin(), end_i = cend(); i != end_i; ++i) {
        out << statKind<T>() << SEP << this->region(i) << SEP
            << this->category(i) << SEP;

        const auto& s = this->stat(i);
        out << StatTotal::str(s.totalTy()) << SEP << s.total();

        out << std::endl;

        if (StatManager::printingThreadVals()) {

          out << statKind<T>() << SEP << this->region(i) << SEP
              << this->category(i) << SEP;
          out << TSTAT_NAME << SEP;

          const char* sep = "";
          for (const auto& v : s.values()) {
            out << sep << v;
            sep = TSTAT_SEP;
          }

          out << std::endl;
        }
      }
    }
  };

  using IntStats     = StatManagerImpl<int64_t>;
  using FPstats      = StatManagerImpl<double>;
  using StrStats     = StatManagerImpl<Str>;
  using int_iterator = typename IntStats::const_iterator;
  using fp_iterator  = typename FPstats::const_iterator;
  using str_iterator = typename StrStats::const_iterator;

  std::string m_outfile;
  IntStats intStats;
  FPstats fpStats;
  StrStats strStats;

protected:
  void mergeStats(void) {
    intStats.mergeStats();
    fpStats.mergeStats();
    strStats.mergeStats();
  }

  int_iterator intBegin(void) const;
  int_iterator intEnd(void) const;

  fp_iterator fpBegin(void) const;
  fp_iterator fpEnd(void) const;

  str_iterator paramBegin(void) const;
  str_iterator paramEnd(void) const;

  template <typename S, typename V>
  void readIntStat(const int_iterator& i, S& region, S& category,
                   int64_t& total, StatTotal::Type& type, V& vec) const {

    intStats.readStat(i, region, category, total, type, vec);
  }

  template <typename S, typename V>
  void readFPstat(const fp_iterator& i, S& region, S& category, double& total,
                  StatTotal::Type& type, V& vec) const {

    fpStats.readStat(i, region, category, total, type, vec);
  }

  template <typename S, typename V>
  void readParam(const str_iterator& i, S& region, S& category, Str& total,
                 StatTotal::Type& type, V& vec) const {

    strStats.readStat(i, region, category, total, type, vec);
  }

  virtual void printStats(std::ostream& out);

  void printHeader(std::ostream& out) const;

public:
  explicit StatManager(const std::string& outfile = "");

  virtual ~StatManager();

  void setStatFile(const std::string& outfile);

  template <typename S1, typename S2, typename T,
            typename = std::enable_if_t<std::is_integral<T>::value ||
                                        std::is_floating_point<T>::value>>
  void addToStat(const S1& region, const S2& category, const T& val,
                 const StatTotal::Type& type) {

    if (std::is_floating_point<T>::value) {
      fpStats.addToStat(gstl::makeStr(region), gstl::makeStr(category),
                        double(val), type);

    } else {
      intStats.addToStat(gstl::makeStr(region), gstl::makeStr(category),
                         int64_t(val), type);
    }
  }

  template <typename S1, typename S2, typename V>
  void addToParam(const S1& region, const S2& category, const V& val) {
    strStats.addToStat(gstl::makeStr(region), gstl::makeStr(category),
                       gstl::makeStr(val), StatTotal::SINGLE);
  }

  void print(void);

private:
};

#else

class StatManager {

  template <typename T>
  struct StatManagerImpl {

    using Str       = galois::gstl::Str;
    using StrSet    = galois::gstl::Set<Str>;
    using Stat      = galois::Accumulator<T>;
    using StatMap   = galois::gstl::Map<std::tuple<Str*, Str*>, Stat*>> ;
    using StatAlloc = galois::FixedSizeAllocator<Stat>;

    StrSet symbols;
    StatMap statMap;
    StatAlloc statAlloc;
    substrate::ThreadRWlock rwmutex;

    ~StatManagerImpl(void) {

      for (auto p : statMap) {
        statAlloc.destruct(p.second);
        statAlloc.deallocate(p.second, 1);
      }
    }

    Stat& getOrInsertMapping(const Str& region, const Str& category) {

      Stat* ret = nullptr;

      auto readAndCheck = [&](void) {
        const Str* ln  = nullptr;
        const Str* cat = nullptr;

        auto ia = symbols.find(region);

        if (ia == symbols.end()) {
          return false; // return early to save a check

        } else {
          ln = &(*ia);
        }

        auto ib = symbols.find(category);
        readOrUpdate

            if (ib == symbols.end()) {
          return false;
        }
        else {
          cat = &(*ib);
        }

        assert(ln && cat);

        auto im = statMap.find(std::make_tuple(ln, cat));
        assert(im != statMap.end() &&
               "statMap lookup shouldn't fail when both symbols exist");

        ret = im->second;

        return true;
      };

      auto write = [&](void) {
        auto p1       = symbols.insert(region);
        const Str* ln = &(p->first);

        auto p2        = symbols.insert(category);
        const Str* cat = &(p->first);

        auto tpl = std::make_tuple(ln, cat);

        Stat* s = statAlloc.allocate(1);
        statAlloc.construct(s);

        auto p = statMap.emplace(tpl, s);
        assert(p.second && "map insert shouldn't fail");

        ret = s;
      };

      galois::substrate::readUpdateProtected(rwmutex, readAndCheck, write);

      assert(ret, "readUpdateProtected shouldn't fail");

      return *ret;
    }

    void addToStat(const Str& region, const Str& category, const T& val) {

      Stat& stat = getOrInserMapping(region, category);
      stat += val;
    }
  };

protected:
  static const char* const SEP = ", ";

  StatManagerImpl<int64_t> intStats;
  StatManagerImpl<double> fpStats;

  void addToStat(const Str& region, const Str& category, int64_t val) {
    intStats.addToStat(region, category, val);
  }

  void addToStat(const Str& region, const Str& category, double val) {
    fpStats.addToStat(region, category, val);
  }
};

#endif

namespace internal {
void setSysStatManager(StatManager* sm);
StatManager* sysStatManager(void);
} // namespace internal

template <typename S1, typename S2, typename T>
inline void reportStat(const S1& region, const S2& category, const T& value,
                       const StatTotal::Type& type) {
  internal::sysStatManager()->addToStat(region, category, value, type);
}

template <typename S1, typename S2, typename T>
inline void reportStat_Single(const S1& region, const S2& category,
                              const T& value) {
  reportStat(region, category, value, StatTotal::SINGLE);
}

template <typename S1, typename S2, typename T>
inline void reportStat_Tmin(const S1& region, const S2& category,
                            const T& value) {
  reportStat(region, category, value, StatTotal::TMIN);
}

template <typename S1, typename S2, typename T>
inline void reportStat_Tmax(const S1& region, const S2& category,
                            const T& value) {
  reportStat(region, category, value, StatTotal::TMAX);
}

template <typename S1, typename S2, typename T>
inline void reportStat_Tsum(const S1& region, const S2& category,
                            const T& value) {
  reportStat(region, category, value, StatTotal::TSUM);
}

template <typename S1, typename S2, typename T>
inline void reportStat_Tavg(const S1& region, const S2& category,
                            const T& value) {
  reportStat(region, category, value, StatTotal::TAVG);
}

template <bool Report = false, typename S1, typename S2, typename T>
inline void reportStatCond(const S1& region, const S2& category, const T& value,
                           const StatTotal::Type& type) {
  if (Report)
    internal::sysStatManager()->addToStat(region, category, value, type);
}

template <bool Report = false, typename S1, typename S2, typename T>
inline void reportStatCond_Single(const S1& region, const S2& category,
                                  const T& value) {
  if (Report)
    reportStat(region, category, value, StatTotal::SINGLE);
}

template <bool Report = false, typename S1, typename S2, typename T>
inline void reportStatCond_Tmin(const S1& region, const S2& category,
                                const T& value) {
  if (Report)
    reportStat(region, category, value, StatTotal::TMIN);
}

template <bool Report = false, typename S1, typename S2, typename T>
inline void reportStatCond_Tmax(const S1& region, const S2& category,
                                const T& value) {
  if (Report)
    reportStat(region, category, value, StatTotal::TMAX);
}

template <bool Report = false, typename S1, typename S2, typename T>
inline void reportStatCond_Tsum(const S1& region, const S2& category,
                                const T& value) {
  if (Report)
    reportStat(region, category, value, StatTotal::TSUM);
}

template <bool Report = false, typename S1, typename S2, typename T>
inline void reportStatCond_Tavg(const S1& region, const S2& category,
                                const T& value) {
  if (Report)
    reportStat(region, category, value, StatTotal::TAVG);
}

template <typename S1, typename S2, typename V>
void reportParam(const S1& region, const S2& category, const V& value) {
  internal::sysStatManager()->addToParam(region, category, value);
}

void setStatFile(const std::string& f);

// TODO: switch to gstl::Str in here
//! Reports Galois system memory stats for all threads
void reportPageAlloc(const char* category);
//! Reports NUMA memory stats for all NUMA nodes
void reportNumaAlloc(const char* category);

} // end namespace runtime
} // end namespace galois

#endif // GALOIS_STAT_MANAGER_H
