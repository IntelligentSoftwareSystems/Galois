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

#ifndef GALOIS_TWOLEVELITERATORA_H
#define GALOIS_TWOLEVELITERATORA_H

#include <cassert>
#include <iterator>
#include <type_traits>
#include <utility>

#include <boost/iterator/iterator_adaptor.hpp>

#include "galois/config.h"
#include "galois/gIO.h"

namespace galois {

/**
 * Alternate implementation of {@link ChooseTwoLevelIterator}.
 */
template <class OuterIter, class InnerIter, class CategoryOrTraversal,
          class InnerBeginFn, class InnerEndFn>
class TwoLevelIteratorA
    : public boost::iterator_adaptor<
          TwoLevelIteratorA<OuterIter, InnerIter, CategoryOrTraversal,
                            InnerBeginFn, InnerEndFn>,
          InnerIter, boost::use_default, CategoryOrTraversal> {
public:
  typedef typename TwoLevelIteratorA::iterator_adaptor_::difference_type
      difference_type;

private:
  OuterIter m_outer_begin; // TODO could skip this field when modeling a forward
                           // iterator
  OuterIter m_outer_end;
  OuterIter m_outer;
  InnerBeginFn m_inner_begin_fn;
  InnerEndFn m_inner_end_fn;

#if __cplusplus >= 201103L
  static_assert(
      std::is_convertible<typename std::result_of<InnerBeginFn(
                              decltype(*std::declval<OuterIter>()))>::type,
                          InnerIter>::value,
      "Result of InnerBeginFn(*OuterIter) should be convertable to InnerIter");
  static_assert(
      std::is_convertible<typename std::result_of<InnerEndFn(
                              decltype(*std::declval<OuterIter>()))>::type,
                          InnerIter>::value,
      "Result of InnerEndFn(*OuterIter) should be convertable to InnerIter");
#endif

  friend class boost::iterator_core_access;

  /**
   * Update base iterator to beginning of first non-empty inner range after
   * current one. Also update outer iterators appropriately.
   */
  void seek_forward() {
    if (this->base_reference() != m_inner_end_fn(*m_outer))
      return;

    ++m_outer;

    for (; m_outer != m_outer_end; ++m_outer) {
      this->base_reference() = m_inner_begin_fn(*m_outer);

      if (this->base_reference() != m_inner_end_fn(*m_outer))
        break;
    }
  }

  template <class Iter>
  void safe_decrement_dispatch(std::forward_iterator_tag, Iter& it,
                               Iter begin) {
    Iter prev = begin;

    for (; begin != it; ++begin)
      prev = begin;
  }

  template <class Iter>
  void safe_decrement_dispatch(std::bidirectional_iterator_tag, Iter& it,
                               const Iter&) {
    --it;
  }

  //! Decrement iterator or return true if it == begin.
  template <class Iter>
  bool safe_decrement(Iter& it, const Iter& begin) {
    if (it == begin)
      return true;
    safe_decrement_dispatch(
        typename std::iterator_traits<Iter>::iterator_category(), it, begin);
    return false;
  }

  template <class Iter>
  typename std::iterator_traits<Iter>::difference_type
  safe_difference_dispatch(Iter it1, Iter it2, Iter end,
                           std::input_iterator_tag) const {
    if (it1 == it2)
      return 0;

    Iter it1_orig(it1);
    Iter it2_orig(it2);

    typename std::iterator_traits<Iter>::difference_type count1 = 0;
    typename std::iterator_traits<Iter>::difference_type count2 = 0;

    while (true) {
      if (it1 != end) {
        ++count1;
        if (++it1 == it2_orig)
          return count1;
      }
      if (it2 != end) {
        ++count2;
        if (++it2 == it1_orig)
          return -count2;
      }
    }
  }

  template <class Iter>
  typename std::iterator_traits<Iter>::difference_type
  safe_difference_dispatch(Iter it1, Iter it2, Iter,
                           std::random_access_iterator_tag) const {
    return std::distance(it1, it2);
  }

  /**
   * Returns correct distances even for forward iterators when it2 is not
   * reachable from it1.
   */
  template <class Iter>
  typename std::iterator_traits<Iter>::difference_type
  safe_distance(Iter it1, Iter it2, Iter end) const {
    return safe_difference_dispatch(
        it1, it2, end,
        typename std::iterator_traits<Iter>::iterator_category());
  }

  /**
   * Update base iterator to end of first non-empty inner range before current
   * one. Also update outer iterators appropriately.
   */
  void seek_backward() {
    InnerIter end;

    for (end = m_inner_end_fn(*m_outer); m_inner_begin_fn(*m_outer) == end;) {
      bool too_far __attribute__((unused)) =
          safe_decrement(m_outer, m_outer_begin);
      assert(!too_far);
      end = m_inner_end_fn(*m_outer);
    }

    this->base_reference() = end;
  }

  void increment() {
    ++this->base_reference();
    seek_forward();
  }

  void decrement() {
    if (m_outer == m_outer_end) {
      bool too_far __attribute__((unused)) =
          safe_decrement(m_outer, m_outer_begin);
      assert(!too_far);
      seek_backward();
    } else if (!safe_decrement(this->base_reference(),
                               m_inner_begin_fn(*m_outer))) {
      // Common case
      return;
    } else {
      // Reached end of inner range
      bool too_far __attribute__((unused)) =
          safe_decrement(m_outer, m_outer_begin);
      assert(!too_far);
      seek_backward();
    }

    bool too_far __attribute__((unused)) =
        safe_decrement(this->base_reference(), m_inner_begin_fn(*m_outer));
    assert(!too_far);
  }

  template <class DiffType = difference_type>
  void advance_dispatch(DiffType n, std::input_iterator_tag) {
    if (n < 0) {
      for (; n; ++n)
        decrement();
    } else if (n > 0) {
      for (; n; --n)
        increment();
    }
  }

  template <class DiffType = difference_type>
  void jump_forward(DiffType n) {
    assert(n >= 0);
    while (n) {
      difference_type k =
          std::distance(this->base_reference(), m_inner_end_fn(*m_outer));
      difference_type m = std::min(k, n);
      n -= m;
      std::advance(this->base_reference(), m);
      if (m == k)
        seek_forward();
    }
  }

  template <class DiffType = difference_type>
  void jump_backward(DiffType n) {
    // Note: not the same as jump_forward due to difference between beginning
    // and end of ranges
    assert(n >= 0);
    if (n && m_outer == m_outer_end) {
      decrement();
      --n;
    }

    while (n) {
      difference_type k =
          std::distance(m_inner_begin_fn(*m_outer), this->base_reference()) + 1;
      if (k == 1) {
        decrement();
        --n;
      } else if (k < n) {
        seek_backward();
        n -= k;
      } else {
        std::advance(this->base_reference(), -n);
        n = 0;
      }
    }
  }

  template <class DiffType = difference_type>
  void advance_dispatch(DiffType n, std::random_access_iterator_tag) {
    if (n == 1)
      increment();
    else if (n == -1)
      decrement();
    else if (n < 0)
      jump_backward(-n);
    else if (n > 0)
      jump_forward(n);
  }

  void advance(difference_type n) {
    advance_dispatch(
        n, typename std::iterator_traits<InnerIter>::iterator_category());
  }

  template <class Other>
  difference_type distance_to_dispatch(Other it2,
                                       std::input_iterator_tag) const {
    // Inline safe_distance here otherwise there is a cyclic dependency:
    // std::distance -> iterator_adaptor -> distance_to -> safe_distance ->
    // std::distance
    if (*this == it2)
      return 0;

    TwoLevelIteratorA it1(*this);
    TwoLevelIteratorA it2_orig(it2);

    difference_type count1 = 0;
    difference_type count2 = 0;

    while (true) {
      if (it1.m_outer != it1.m_outer_end) {
        ++count1;
        if (++it1 == it2_orig)
          return count1;
      }
      if (it2.m_outer != it2.m_outer_end) {
        ++count2;
        if (++it2 == *this)
          return -count2;
      }
    }
  }

  template <class Other>
  difference_type distance_to_dispatch(const Other& x,
                                       std::random_access_iterator_tag) const {
    if (*this == x)
      return 0;
    else if (m_outer == x.m_outer)
      return safe_distance(this->base_reference(), x.base_reference(),
                           m_inner_end_fn(*m_outer));
    else if (safe_distance(m_outer, x.m_outer, m_outer_end) < 0)
      return -x.distance_to(*this);

    difference_type me_count = 0;

    TwoLevelIteratorA me(*this);

    while (me.m_outer != me.m_outer_end) {
      difference_type d;
      if (me.m_outer != x.m_outer)
        d = std::distance(me.base_reference(), me.m_inner_end_fn(*me.m_outer));
      else
        d = std::distance(me.base_reference(), x.base_reference());
      me_count += d;
      std::advance(me, d);
      if (me == x)
        return me_count;
    }

    GALOIS_DIE("invalid iterator ", std::distance(m_outer, x.m_outer));
    return 0;
  }

  template <class OtherOuterIter, class OtherInnerIter, class C, class BF,
            class EF>
  difference_type distance_to(
      const TwoLevelIteratorA<OtherOuterIter, OtherInnerIter, C, BF, EF>& x)
      const {
    return distance_to_dispatch(
        x, typename std::iterator_traits<InnerIter>::iterator_category());
  }

  template <class OtherOuterIter, class OtherInnerIter, class C, class BF,
            class EF>
  bool
  equal(const TwoLevelIteratorA<OtherOuterIter, OtherInnerIter, C, BF, EF>& x)
      const {
    if (m_outer == m_outer_end && m_outer == x.m_outer)
      return true;

    return m_outer == x.m_outer && this->base_reference() == x.base_reference();
  }

public:
  TwoLevelIteratorA() {}

  TwoLevelIteratorA(OuterIter outer_begin, OuterIter outer_end, OuterIter outer,
                    InnerBeginFn inner_begin_fn, InnerEndFn inner_end_fn)
      : m_outer_begin(outer_begin), m_outer_end(outer_end), m_outer(outer),
        m_inner_begin_fn(inner_begin_fn), m_inner_end_fn(inner_end_fn) {
    if (m_outer != m_outer_end) {
      this->base_reference() = m_inner_begin_fn(*m_outer);
      seek_forward();
    }
  }

  TwoLevelIteratorA(OuterIter outer_begin, OuterIter outer_end, OuterIter outer,
                    InnerIter inner, InnerBeginFn inner_begin_fn,
                    InnerEndFn inner_end_fn)
      : m_outer_begin(outer_begin), m_outer_end(outer_end), m_outer(outer),
        m_inner_begin_fn(inner_begin_fn), m_inner_end_fn(inner_end_fn) {
    this->base_reference() = inner;
  }

  const OuterIter& get_outer_reference() const { return m_outer; }

  const InnerIter& get_inner_reference() const {
    return this->base_reference();
  }
};

//! Helper functor, returns <code>t.end()</code>
struct GetBegin {
  template <class T>
  auto operator()(T&& x) const -> decltype(std::forward<T>(x).begin()) {
    return std::forward<T>(x).begin();
  }
};

//! Helper functor, returns <code>t.end()</code>
struct GetEnd {
  template <class T>
  auto operator()(T&& x) const -> decltype(std::forward<T>(x).end()) {
    return std::forward<T>(x).end();
  }
};

#if __cplusplus >= 201103L
template <
    class CategoryOrTraversal = std::forward_iterator_tag, class OuterIter,
    class InnerIter           = decltype(std::declval<OuterIter>()->begin()),
    class InnerBeginFn = GetBegin, class InnerEndFn = GetEnd,
    class Iter = TwoLevelIteratorA<OuterIter, InnerIter, CategoryOrTraversal,
                                   InnerBeginFn, InnerEndFn>>
std::pair<Iter, Iter> make_two_level_iterator(OuterIter outer_begin,
                                              OuterIter outer_end) {
  return std::make_pair(
      Iter(outer_begin, outer_end, outer_begin, InnerBeginFn(), InnerEndFn()),
      Iter(outer_begin, outer_end, outer_end, InnerBeginFn(), InnerEndFn()));
}
#else
// XXX(ddn): More direct encoding crashes XL 12.1, so lean towards more verbose
// types
template <class CategoryOrTraversal, class OuterIter, class InnerIter,
          class InnerBeginFn, class InnerEndFn>
std::pair<TwoLevelIteratorA<OuterIter, InnerIter, CategoryOrTraversal,
                            InnerBeginFn, InnerEndFn>,
          TwoLevelIteratorA<OuterIter, InnerIter, CategoryOrTraversal,
                            InnerBeginFn, InnerEndFn>>
make_two_level_iterator(OuterIter outer_begin, OuterIter outer_end) {
  return std::make_pair(
      TwoLevelIteratorA<OuterIter, InnerIter, CategoryOrTraversal, InnerBeginFn,
                        InnerEndFn>(outer_begin, outer_end, outer_begin,
                                    InnerBeginFn(), InnerEndFn()),
      TwoLevelIteratorA<OuterIter, InnerIter, CategoryOrTraversal, InnerBeginFn,
                        InnerEndFn>(outer_begin, outer_end, outer_end,
                                    InnerBeginFn(), InnerEndFn()));
}
#endif

} // end namespace galois

#endif
