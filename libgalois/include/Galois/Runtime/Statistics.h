#ifndef GALOIS_STAT_MANAGER_H
#define GALOIS_STAT_MANAGER_H

#include "Galois/gstl.h"
#include "Galois/Substrate/PerThreadStorage.h"
#include "Galois/Substrate/ThreadRWlock.h"


#include <limits>
/**
 * TODO: 
 * Print intra host stats with per-thread details and inter-host stats with per-host details
 * print to 2 files if supporting R format
 * libdist implements an addToStat with host ID and manages inter-host stats and their combining

 */

namespace Galois {
namespace Runtime {

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


#define STAT_MANAGER_IMPL 2// 0 or 1 or 2

#if STAT_MANAGER_IMPL == 0

class StatManager {

  using Str = Galois::gstl::Str;
  using StrSet = Galois::gstl::Set<Str>;
  using StatMap = Galois::gstl::Map<std::tuple<const Str*, const Str*>, T>;


  template <typename T>
  struct ThreadStatManager {

    StrSet symbols;
    StatMap statMap;

    const Str* getOrInsertSymbol(const Str& s) {
      auto i = symbols.insert(s);
      return &(i->first);
    }

    const Str* getSymbol(const Str& s) const {
      auto i = symbols.find(s);

      if (i == symbols.cend()) {
        return nullptr;
      } else {
        return &(*i);
      }
    }

    void addToStat(const Str& loopname, const Str& category, const T& val) {

      const Str* ln = getOrInsertSymbol(loopname);
      const Str* cat = getOrInsertSymbol(category);

      auto tpl = std::make_tuple(ln, cat);

      auto p = statMap.emplace(tpl, val);

      if (!p.second) {
        p.first.second += val;
      }
    }
  };


  template <typename T>
  struct StatManagerImpl {

    Substrate::PerThreadStorage<ThreadStatManager<T>> perThreadManagers;
    ThreadStatManager result;
    bool merged = false;

    
    void addToStat(const Str& loopname, const Str& category, const T& val) {
      perThreadManagers.getLocal()->addToStat(loopname, category, val);
    }

    void mergeStats(void) {

      if (merged) { return; }

      for (unsigned t = 0; t < perThreadManagers.size(); ++t) {

        ThreadStatManager* manager = *perThreadManagers.getRemote(t);

        for (auto p: manager->statMap) {
          result.addToStat(*std::get<0>(p.first), *std::get<1>(p.first), p.second);
        }
      }

      merged = true;
    }


  };


  StatManagerImpl<int64_t> intStats;
  StatManagerImpl<double> fpStats;

protected:


public:


  void addToStat(const Str& loopname, const Str& category, int64_t val) {
    intStats.addToStat(loopname, category, val);
  }

  void addToStat(const Str& loopname, const Str& category, double val) {
    fpStats.addToStat(loopname, category, val);
  }

  void print() {
    printHeader();
    intStats.print();
    fpStats.print();
  }

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


} // end namespace Runtime
} // end namespace Galois


#endif// GALOIS_STAT_MANAGER_H
