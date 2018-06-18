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

#ifndef GALOIS_GSTL_H
#define GALOIS_GSTL_H

#include "galois/PriorityQueue.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <cassert>
#include <vector>
#include <set>
#include <deque>
#include <map>
#include <list>
#include <string>
#include <sstream>

namespace galois {

namespace gstl {

//! [define Pow_2_VarSizeAlloc]
template <typename T>
using Pow2Alloc = typename runtime::Pow_2_BlockAllocator<T>;
//! [define Pow_2_VarSizeAlloc]

template <typename T>
using FixedSizeAlloc = typename runtime::FixedSizeAllocator<T>;

//! [STL vector using Pow_2_VarSizeAlloc]
template <typename T>
using Vector = std::vector<T, Pow2Alloc<T>>;
//! [STL vector using Pow_2_VarSizeAlloc]

template <typename T>
using Deque = std::deque<T, Pow2Alloc<T>>;

template <typename T>
using List = std::list<T, FixedSizeAlloc<T>>;

template <typename T, typename C = std::less<T>>
using Set = std::set<T, C, FixedSizeAlloc<T>>;

template <typename K, typename V, typename C = std::less<K>>
using Map = std::map<K, V, C, FixedSizeAlloc<std::pair<const K, V>>>;

template <typename T, typename C = std::less<T>>
using PQ = MinHeap<T, C, Vector<T>>;

using Str = std::basic_string<char, std::char_traits<char>, Pow2Alloc<char>>;

template <typename T>
struct StrMaker {
  Str operator()(const T& x) const {
    std::basic_ostringstream<char, std::char_traits<char>, Pow2Alloc<char>> os;
    os << x;
    return Str(os.str());
  }
};

template <>
struct StrMaker<std::string> {
  Str operator()(const std::string& x) const { return Str(x.begin(), x.end()); }
};

template <>
struct StrMaker<Str> {
  const Str& operator()(const Str& x) const { return x; }
};

template <>
struct StrMaker<const char*> {
  Str operator()(const char* x) const { return Str(x); }
};

template <typename T>
Str makeStr(const T& x) {
  return StrMaker<T>()(x);
}
} // end namespace gstl

template <typename I>
class IterRange {
  I m_beg;
  I m_end;

public:
  IterRange(const I& b, const I& e) : m_beg(b), m_end(e) {}
  const I& begin(void) const { return m_beg; }
  const I& end(void) const { return m_end; }
};

template <typename I>
auto makeIterRange(const I& beg, const I& end) {
  return IterRange<I>(beg, end);
}

template <typename C>
auto makeIterRange(C&& cont) {
  using I = decltype(std::forward<C>(cont).begin());
  return IterRange<I>(std::forward<C>(cont).begin(),
                      std::forward<C>(cont).end());
}

namespace internal {

template <typename T, typename C>
struct SerCont {
  C m_q;

  explicit SerCont(const C& q = C()) : m_q(q) {}

  void push(const T& x) { m_q.push_back(x); }

  template <typename I>
  void push(const I& beg, const I& end) {
    for (I i = beg; i != end; ++i) {
      push(*i);
    }
  }

  template <typename... Args>
  void emplace(Args&&... args) {
    m_q.emplace_back(std::forward<Args>(args)...);
  }

  bool empty(void) const { return m_q.empty(); }

  void clear(void) { m_q.clear(); }

  using value_type     = typename C::value_type;
  using iterator       = typename C::iterator;
  using const_iterator = typename C::const_iterator;

  iterator begin(void) { return m_q.begin(); }
  iterator end(void) { return m_q.end(); }

  const_iterator begin(void) const { return m_q.begin(); }
  const_iterator end(void) const { return m_q.end(); }

  const_iterator cbegin(void) const { return m_q.cbegin(); }
  const_iterator cend(void) const { return m_q.cend(); }
};
} // namespace internal

template <typename T, typename C = std::deque<T>>
class SerFIFO : public internal::SerCont<T, C> {

  using Base = internal::SerCont<T, C>;

public:
  explicit SerFIFO(const C& q = C()) : Base(q) {}

  T pop(void) {
    T ret = Base::m_q.front();
    Base::m_q.pop_front();
    return ret;
  }
};

template <typename T, typename C = std::vector<T>>
class SerStack : public internal::SerCont<T, C> {

  using Base = internal::SerCont<T, C>;

public:
  explicit SerStack(const C& q = C()) : Base(q) {}

  T pop(void) {
    T ret = Base::m_q.back();
    Base::m_q.pop_back();
    return ret;
  }
};

template <typename IterTy, class Distance>
IterTy safe_advance_dispatch(IterTy b, IterTy e, Distance n,
                             std::random_access_iterator_tag) {
  if (std::distance(b, e) >= n)
    return b + n;
  else
    return e;
}

template <typename IterTy, class Distance>
IterTy safe_advance_dispatch(IterTy b, IterTy e, Distance n,
                             std::input_iterator_tag) {
  while (b != e && n--)
    ++b;
  return b;
}

/**
 * Like std::advance but returns end if end is closer than the advance amount.
 */
template <typename IterTy, class Distance>
IterTy safe_advance(IterTy b, IterTy e, Distance n) {
  typename std::iterator_traits<IterTy>::iterator_category category;
  return safe_advance_dispatch(b, e, n, category);
}

/**
 * Finds the midpoint of a range.  The first half is always be bigger than
 * the second half if the range has an odd length.
 */
template <typename IterTy>
IterTy split_range(IterTy b, IterTy e) {
  std::advance(b, (std::distance(b, e) + 1) / 2);
  return b;
}

/**
 * Returns a continuous block from the range based on the number of
 * divisions and the id of the block requested
 */
template <
    typename IterTy,
    typename std::enable_if<!std::is_integral<IterTy>::value>::type* = nullptr>
std::pair<IterTy, IterTy> block_range(IterTy b, IterTy e, unsigned id,
                                      unsigned num) {
  size_t dist = std::distance(b, e);
  size_t numper = std::max((dist + num - 1) / num, (size_t)1); // round up
  size_t A      = std::min(numper * id, dist);
  size_t B      = std::min(numper * (id + 1), dist);
  std::advance(b, A);

  if (dist != B) {
    e = b;
    std::advance(e, B - A);
  }

  return std::make_pair(b, e);
}

template <typename IntTy, typename std::enable_if<
                              std::is_integral<IntTy>::value>::type* = nullptr>
std::pair<IntTy, IntTy> block_range(IntTy b, IntTy e, unsigned id,
                                    unsigned num) {
  IntTy dist   = e - b;
  IntTy numper = std::max((dist + num - 1) / num, (IntTy)1); // round up
  IntTy A      = std::min(numper * id, dist);
  IntTy B      = std::min(numper * (id + 1), dist);
  b += A;
  if (dist != B) {
    e = b;
    e += (B - A);
  }
  return std::make_pair(b, e);
}

namespace internal {
template <typename I>
using Val_ty = typename std::iterator_traits<I>::value_type;
} // namespace internal

//! Destroy a range
template <typename I>
std::enable_if_t<!std::is_scalar<internal::Val_ty<I>>::value>
uninitialized_destroy(I first, I last) {

  using T = internal::Val_ty<I>;
  for (; first != last; ++first)
    (&*first)->~T();
}

template <class I>
std::enable_if_t<std::is_scalar<internal::Val_ty<I>>::value>
uninitialized_destroy(I, I) {}

} // namespace galois
#endif
