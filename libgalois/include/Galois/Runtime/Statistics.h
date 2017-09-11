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

#ifndef GALOIS_STAT_MANAGER_H
#define GALOIS_STAT_MANAGER_H

#include "Galois/gstl.h"
#include "Galois/gIO.h"
#include "Galois/Substrate/PerThreadStorage.h"
#include "Galois/Substrate/ThreadRWlock.h"

#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.

#include <limits>
#include <string>
#include <map>
/**
 * TODO: 
 * Print intra host stats with per-thread details and inter-host stats with per-host details
 * print to 2 files if supporting R format
 * libdist implements an addToStat with host ID and manages inter-host stats and their combining

 */

namespace Galois {
namespace Runtime {

boost::uuids::uuid getRandUUID();

template <typename T>
class RunningMin {
  T m_min;

public:

  RunningMin(void): m_min(std::numeric_limits<T>::max()) {
  }

  void add(const T& val) {
    m_min = std::min(m_min, val);
  }

  const T& min(void) const { return m_min; }

};


template <typename T>
class RunningMax {
  T m_max;

public:

  RunningMax(void): m_max(std::numeric_limits<T>::min()) {
  }

  void add(const T& val) {
    m_max = std::max(m_max, val);
  }

  const T& max(void) const { return m_max; }

};

template <typename T>
class RunningSum {
  T m_sum;
  size_t m_count;

public:

  RunningSum(void): m_sum(), m_count(0) {
  }

  void add(const T& val) {
    m_sum += val;
    ++m_count;
  }

  const T& sum(void) const { return m_sum; }

  const size_t& count(void) const { return m_count; }

  T avg () const { return m_sum / T (m_count); }

};

template <typename T>
class RunningVec {

  using Vec = gstl::Vector<T>

  Vec m_vec;

public:


  void add(const T& val) {
    m_vec.push_back(val);
  }

  const Vec& values(void) const { return m_vec; }
};

template <typename T>
class NamedStat {

  using Str = Galois::stl::Str;

  Str m_name;

public:

  void setName(const Str& name) {
    m_name = name;
  }

  void setName(Str&& name) {
    m_name = std::move(name);
  }

  const Str& name(void) const {
    return m_name;
  }

  void add(const T& val) const {}
};

template <typename T, typename... Bases>
class AggStatistic: public Bases... {

public:

  using with_min = AggStatistic<T, RunningMin<T>, Bases...>;

  using with_max = AggStatistic<T, RunningMax<T>, Bases...>;

  using with_sum = AggStatistic<T, RunningSum<T>, Bases...>;

  using with_mem = AggStatistic<T, RunningVec<T>, Bases...>;

  using with_name = AggStatistic<T, NamedStat<T>, Bases...>;


  void add(const T& val) {
    using Expander = int[];

    (void) Expander {0, ( (void) Base::add(val), 0)...};
  }

};


namespace hidden {

template <typename Stat>
struct BasicStatMap {

  using Str = Galois::gstl::Str;
  using StrSet = Galois::gstl::Set<Str>;
  using StatMap = Galois::gstl::Map<std::tuple<const Str*, const Str*>, Stat>;
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

  Stat& getOrInsertStat(const Str& loopname, const Str& category) {
    
    const Str* ln = getOrInsertSymbol(loopname);
    const Str* cat = getOrInsertSymbol(category);

    auto tpl = std::make_tpl(ln, cat);

    auto p = statMap.emplace(tpl, Stat());

    return p.first->second;
  }

  template <typename T>
  void addToStat(const Str& loopname, const Str& category, const T& val) {
    Stat& s = getOrInsertStat(loopname, category);
    s.add(val);
  }

  const_iterator begin(void) const { return statMap.begin(); } 
  const_iterator end(void) const { return statMap.end(); } 

  const_iterator cbegin(void) const { return statMap.cbegin(); } 
  const_iterator cend(void) const { return statMap.cend(); } 

  const Str& name(const const_iterator& i) const { return *(std::get<0>(i->first)); }

  const Str& category(const const_iterator& i) const { return *(std::get<1>(i->first)); }

  const Stat& stat(const const_iterator& i) const { return i->second; }

};

template <typename T>
using VecStat_with_MinMaxSum = typename AggStatistic<T>::with_mem::with_min::with_max::with_sum;


template <typename T>
using VecStatManager = BasicStatMap<VecStat_with_MinMaxSum<T> >;

template <typename T>
struct ScalarStat {
  T m_val;

  ScalarStat(void): m_val() {}

  void add (const T& v) {
    m_val += v;
  }

  operator const T& (void) const { return m_val; }

};

template <typename T>
using ScalarStatManager = BasicStatMap<ScalarStat<T> >;


} // end namespace hidden

#define STAT_MANAGER_IMPL 0// 0 or 1 or 2

#if STAT_MANAGER_IMPL == 0

class StatManager {

  using Str = Galois::gstl::Str;

  template <typename T>
  struct StatManagerImpl {

    using MergedStats = hidden::VecStatManager<T>;
    using const_iterator = typename MergedStats::const_iterator;

    Substrate::PerThreadStorage<hidden::ScalarStatManager<T> > perThreadManagers;
    MergedStats result;
    bool merged = false;

    
    void addToStat(const Str& loopname, const Str& category, const T& val) {
      perThreadManagers.getLocal()->addToStat(loopname, category, val);
    }

    void mergeStats(void) {

      if (merged) { return; }

      for (unsigned t = 0; t < perThreadManagers.size(); ++t) {

        ThreadStatManager* manager = *perThreadManagers.getRemote(t);

        for (auto i = manager->cbegin(), end_i = manager.cend(); i != end_i; ++i) {
          result.addToStat(manager->name(i), manager->category(i), T(manager->stat(i)));
        }
      }

      merged = true;
    }


    const_iterator begin(void) const { return result.begin(); } 
    const_iterator end(void) const { return result.end(); } 

    const_iterator cbegin(void) const { return result.cbegin(); } 
    const_iterator cend(void) const { return result.cend(); } 

    const Str& name(const const_iterator& i) const { return result.name(i); }

    const Str& category(const const_iterator& i) const { return result.category(i);  }

    const Stat& stat(const const_iterator& i) const { return result.stat(i); }

    template <typename S, typename V>
    void readStat(const const_iterator& i, S& name, S& category, V& vec) const {
      name = this->name(i);
      category = this->category(i);

      vec.clear();
      vec = this->stat(i).values();
    }

    void print(std::ostream& out) const {
      for (auto i = cbegin(), end_i = cend(); i != end_i; ++i) {
        out << "STAT" << SEP << this->name(i) << SEP << this->category(i) << SEP;

        const auto& s = this->stat(i);
        out << s.sum() << SEP << s.avg() << SEP << s.min() << SEP << s.max();

        for (const auto i: s.values()) {
          out << SEP << i;
        }

        out << std::endl;
      }
    }

  };

  using IntStats = StatManagerImpl<int64_t>;
  using FPstats = StatManagerImpl<double>;
  using int_iterator = typename IntStats::const_iterator;
  using fp_iterator = typename FPstats::const_iterator;

  static const char* const SEP = ", ";

  Str m_outfile;
  IntStats intStats;
  FPstats fpStats;

protected:

  void mergeStats(void) {
    intStats.mergeStats();
    fpStats.mergeStats();
  }

  int_iterator intBegin(void) const;
  int_iterator intEnd(void) const;

  fp_iterator fpBegin(void) const;
  fp_iterator fpEnd(void) const;

  template <typename S, typename V>
  void readIntStat(const typename IntStats::const_iterator& i, S& name, S& category, V& vec) const {
    intStats.readStat(i, name, category, vec);
  }

  template <typename S, typename V>
  void readFPstat(const typename IntStats::const_iterator& i, S& name, S& category, V& vec) const {
    fpStats.readStat(i, name, category, vec);
  }

  virtual void printStats(std::ostream& out);

  virtual void printHeader(std::ostream& out);

  unsigned maxThreads(void) const;

public:

  explicit StatCollector(const Str& outfile="");

  void addToStat(const Str& loopname, const Str& category, int64_t val) {
    intStats.addToStat(loopname, category, val);
  }

  void addToStat(const Str& loopname, const Str& category, double val) {
    fpStats.addToStat(loopname, category, val);
  }

  void print(void);

};

#elif STAT_MANAGER_IMPL == 1

class StatManager {

  using Str = Galois::gstl::Str;
  using StrSet = Galois::gstl::Set<Str>;
  using Stat = Galois::Accumulator<int64_t>;
  using StatMap = Galois::gstl::Map<std::tuple<Str*, Str*>, Stat*>;
  using StatAlloc = Galois::FixedSizeAllocator<Stat>;

  StrSet symbols;
  StatMap statMap;
  StatAlloc statAlloc;

public:

  void addToStat(const Str& loopname, const Str& category, int64_t val) {

    auto* accum = getOrInsertMapping(loopname, category);

    *accum += val;


    Str* ln = symbols.getOrInsert(loopname);
    Str* cat = symbols.getOrInsert(category);

    auto tpl = std::make_tuple(ln, cat);

    auto stat = statMap.getMapping(tpl);

    if (!stat) {
      Stat* stat = statAlloc.allocate(1);
      statAlloc.construct(stat);

      auto p = statMap.getOrInsertMapping(tpl, stat);

      *(p.first) += val;

      if (!p.second) {
        statAlloc.destruct(stat);
        statAlloc.deallocate(stat, 1);
      } 
    }
  }

protected:



};

#else

class StatManager {

  template<typename T> 
  struct StatManagerImpl {

    using Str = Galois::gstl::Str;
    using StrSet = Galois::gstl::Set<Str>;
    using Stat = Galois::Accumulator<T>;
    using StatMap = Galois::gstl::Map<std::tuple<Str*, Str*>, Stat*> >;
    using StatAlloc = Galois::FixedSizeAllocator<Stat>;

    StrSet symbols;
    StatMap statMap;
    StatAlloc statAlloc;
    Substrate::ThreadRWlock rwmutex;

    ~StatManagerImpl(void) {

      for (auto p: statMap) {
        statAlloc.destruct(p.second);
        statAlloc.deallocate(p.second, 1);
      }
    }

    Stat& getOrInsertMapping(const Str& loopname, const Str& category) {

      Stat* ret = nullptr;

      auto readAndCheck = [&] (void) {

        const Str* ln = nullptr;
        const Str* cat = nullptr;
        
        auto ia = symbols.find(loopname);

        if (ia == symbols.end()) {
          return false; // return early to save a check

        } else {
          ln = &(*ia);
        }

        auto ib = symbols.find(category);readOrUpdate

        if (ib == symbols.end()) {
          return false;

        } else {
          cat = &(*ib);
        }

        assert(ln && cat);

        auto im = statMap.find(std::make_tuple(ln, cat));
        assert(im != statMap.end() && "statMap lookup shouldn't fail when both symbols exist");

        ret = im->second;

        return true;

      };

      auto write = [&] (void) {
        auto p1 = symbols.insert(loopname);
        const Str* ln = &(p->first);

        auto p2 = symbols.insert(category);
        const Str* cat = &(p->first);

        auto tpl = std::make_tuple(ln, cat);

        Stat* s = statAlloc.allocate(1);
        statAlloc.construct(s);

        auto p = statMap.emplace(tpl, s);
        assert(p.second && "map insert shouldn't fail");
        
        ret = s;
      };

      Galois::Substrate::readUpdateProtected(rwmutex, readAndCheck, write);

      assert(ret, "readUpdateProtected shouldn't fail");

      return *ret;

    }

    void addToStat(const Str& loopname, const Str& category, const T& val) {

      Stat& stat = getOrInserMapping(loopname, category);
      stat += val;

    }
  };

protected:

  static const char* const SEP = ", ";

  StatManagerImpl<int64_t> intStats;
  StatManagerImpl<double> fpStats;

  void addToStat(const Str& loopname, const Str& category, int64_t val) {
    intStats.addToStat(loopname, category, val);
  }

  void addToStat(const Str& loopname, const Str& category, double val) {
    fpStats.addToStat(loopname, category, val);
  }

};

#endif

namespace internal {
  void setSysStatManager(StatManager* sm);
  StatManager* sysStatManager(void);
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T> > >
void reportStat(const char* loopname, const char* category, const T& value) {
  sysStatManager()->addToStat(
      gst::String(loopname? loopname: "(NULL)"),
      gst::String(category? category: "(NULL)"),
      int64_t(value));
}

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T> > >
void reportStat(const char* loopname, const char* category, const T& value) {
  sysStatManager()->addToStat(
      gst::String(loopname? loopname: "(NULL)"),
      gst::String(category? category: "(NULL)"),
      double(value));
}


//! Reports stats for a given thread
void reportParam(const char* loopname, const char* category, const std::string& value);

template <typename T, typename = std::enable_if_t<std::is_integral_v<T> > >
void reportStat(const gstl::String& loopname, const gstl::String& category, const T& value) {
  sysStatManager()->addToStat(loopname, category, int64_t(value));
}

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T> > >
void reportStat(const gstl::String& loopname, const gstl::String& category, const T& value) {
  sysStatManager()->addToStat(loopname, category, double(value));
}

void reportParam(const gstl::String& loopname, const gstl::String& category, const gstl::String& value);

//! Reports Galois system memory stats for all threads
void reportPageAlloc(const char* category);
//! Reports NUMA memory stats for all NUMA nodes
void reportNumaAlloc(const char* category);


} // end namespace Runtime
} // end namespace Galois


#endif// GALOIS_STAT_MANAGER_H
