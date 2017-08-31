#ifndef GALOIS_AGG_STATS_H
#define GALOIS_AGG_STATS_H

#include "Galois/gstl.h"

#include <limits>

namespace Galois {

template <typename T>
class RunningMin {
  T m_min;

public:

  RunningMin(void): m_min(std::numeric_limits<T>::max()) {
  }

  void add(const T& val) {
    m_min = std::min(m_min, val);
  }

  const T& getMin(void) const { return m_min; }

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

  const T& getMax(void) const { return m_max; }

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

  const T& getSum(void) const { return m_sum; }

  const size_t& getCount(void) const { return m_count; }

  T getAvg () const { return m_sum / T (m_count); }

};

template <typename T>
class RunningVec {

  using Vec = Galois::Vector<T>;

  Vec m_vec;

public:


  void add(const T& val) {
    m_vec.emplace_back(val);
  }

  const Vec& getVec(void) const { return m_vec; }
};

template <typename T, typename... Bases>
class AggStats: public Bases... {

  const char* const m_name;

public:

  using with_min = AggStats<T, RunningMin<T>, Bases...>;

  using with_max = AggStats<T, RunningMax<T>, Bases...>;

  using with_sum = AggStats<T, RunningSum<T>, Bases...>;

  using with_mem = AggStats<T, RunningVec<T>, Bases...>;

  explicit AggStats(const char* const name): Bases()..., m_name(name) 
  {
    if (m_name == nullptr) {
      m_name = "AGG_STAT";
    }
  } 

  const char* const getName(void) const { return m_name; }

  void add(const T& val) {
    using Expander = int[];

    (void) Expander {0, ( (void) Base::add(val), 0)...};
  }

};


} // end namespace Galois
#endif// GALOIS_AGG_STATS_H
