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

#ifndef GALOIS_RUNTIME_RANGE_H
#define GALOIS_RUNTIME_RANGE_H

#include <iterator>

#include <boost/iterator/counting_iterator.hpp>

#include "galois/config.h"
#include "galois/gstl.h"
#include "galois/substrate/ThreadPool.h"

namespace galois {
namespace runtime {

extern unsigned int activeThreads;

// TODO(ddn): update to have better forward iterator behavor for blocked/local
// iteration

template <typename T>
class LocalRange {
  T* container;

public:
  typedef T container_type;
  typedef typename T::iterator iterator;
  typedef typename T::local_iterator local_iterator;
  typedef iterator block_iterator;
  typedef typename std::iterator_traits<iterator>::value_type value_type;

  LocalRange(T& c) : container(&c) {}

  iterator begin() const { return container->begin(); }
  iterator end() const { return container->end(); }

  // TODO fix constness of local containers
  /* const */ T& get_container() const { return *container; }

  std::pair<block_iterator, block_iterator> block_pair() const {
    return galois::block_range(begin(), end(), substrate::ThreadPool::getTID(),
                               activeThreads);
  }

  std::pair<local_iterator, local_iterator> local_pair() const {
    return std::make_pair(container->local_begin(), container->local_end());
  }

  local_iterator local_begin() const { return container->local_begin(); }
  local_iterator local_end() const { return container->local_end(); }

  block_iterator block_begin() const { return block_pair().first; }
  block_iterator block_end() const { return block_pair().second; }
};

template <typename T>
inline LocalRange<T> makeLocalRange(T& obj) {
  return LocalRange<T>(obj);
}

template <typename IterTy>
class StandardRange {
  IterTy ii, ei;

public:
  typedef IterTy iterator;
  typedef iterator local_iterator;
  typedef iterator block_iterator;

  typedef typename std::iterator_traits<IterTy>::value_type value_type;

  StandardRange(IterTy b, IterTy e) : ii(b), ei(e) {}

  iterator begin() const { return ii; }
  iterator end() const { return ei; }

  std::pair<block_iterator, block_iterator> block_pair() const {
    return galois::block_range(ii, ei, substrate::ThreadPool::getTID(),
                               activeThreads);
  }

  std::pair<local_iterator, local_iterator> local_pair() const {
    return block_pair();
  }

  local_iterator local_begin() const { return block_begin(); }
  local_iterator local_end() const { return block_end(); }

  block_iterator block_begin() const { return block_pair().first; }
  block_iterator block_end() const { return block_pair().second; }
};

template <typename IterTy>
inline StandardRange<IterTy> makeStandardRange(IterTy begin, IterTy end) {
  return StandardRange<IterTy>(begin, end);
}

/**
 * SpecificRange is a range type where a threads range is specified by
 * an an int array that tells you where each thread should begin its
 * iteration
 */
template <typename IterTy>
class SpecificRange {
  IterTy global_begin, global_end;
  const uint32_t* thread_beginnings;

public:
  typedef IterTy iterator;
  typedef iterator local_iterator;
  typedef iterator block_iterator;

  typedef typename std::iterator_traits<IterTy>::value_type value_type;

  SpecificRange(IterTy b, IterTy e, const uint32_t* thread_ranges)
      : global_begin(b), global_end(e), thread_beginnings(thread_ranges) {}

  iterator begin() const { return global_begin; }
  iterator end() const { return global_end; }

  /* Using the thread_beginnings array which tells you which node each thread
   * should begin at, we can get the local block range for a particular
   * thread. If the local range falls outside of global range, do nothing.
   *
   * @returns A pair of iterators that specifies the beginning and end
   * of the range for this particular thread.
   */
  std::pair<block_iterator, block_iterator> block_pair() const {
    uint32_t my_thread_id  = substrate::ThreadPool::getTID();
    uint32_t total_threads = runtime::activeThreads;

    iterator local_begin = thread_beginnings[my_thread_id];
    iterator local_end   = thread_beginnings[my_thread_id + 1];

    assert(local_begin <= local_end);

    if (thread_beginnings[total_threads] == *global_end && *global_begin == 0) {
      return std::make_pair(local_begin, local_end);
    } else {
      // This path assumes that we were passed in thread_beginnings for the
      // range 0 to last node, but the passed in range to execute is NOT the
      // entire 0 to thread end range; therefore, work under the assumption that
      // only some threads will execute things only if they "own" nodes in the
      // range
      iterator left  = local_begin;
      iterator right = local_end;

      // local = what this thread CAN do
      // global = what this thread NEEDS to do

      // cutoff left and right if global begin/end require less than what we
      // need
      if (local_begin < global_begin) {
        left = global_begin;
      }
      if (local_end > global_end) {
        right = global_end;
      }
      // make sure range is sensible after changing left and right
      if (left >= right || right <= left) {
        left = right = global_end;
      }

      // Explanations/reasoning of possible cases
      // [ ] = local ranges
      // o = need to be included; global ranges = leftmost and rightmost circle
      // x = not included
      // ooooo[ooooooooxxxx]xxxxxx handled (left the same, right moved)
      // xxxxx[xxxxxooooooo]oooooo handled (left moved, right the same)
      // xxxxx[xxoooooooxxx]xxxxxx handled (both left/right moved)
      // xxxxx[xxxxxxxxxxxx]oooooo handled (left will be >= right, set l = r)
      // oooox[xxxxxxxxxxxx]xxxxxx handled (right will be <= left, set l = r)
      // xxxxx[oooooooooooo]xxxxxx handled (left, right the same = local range)

      return std::make_pair(left, right);
    }
  }

  std::pair<local_iterator, local_iterator> local_pair() const {
    return block_pair();
  }

  local_iterator local_begin() const { return block_begin(); }
  local_iterator local_end() const { return block_end(); }

  block_iterator block_begin() const { return block_pair().first; }
  block_iterator block_end() const { return block_pair().second; }
};

/**
 * Creates a SpecificRange object.
 *
 * @tparam IterTy The iterator type used by the range object
 * @param begin The global beginning of the range
 * @param end The global end of the range
 * @param thread_ranges An array of iterators that specifies where each
 * thread's range begins
 * @returns A SpecificRange object
 */
template <typename IterTy>
inline SpecificRange<IterTy> makeSpecificRange(IterTy begin, IterTy end,
                                               const uint32_t* thread_ranges) {
  return SpecificRange<IterTy>(begin, end, thread_ranges);
}

} // end namespace runtime

namespace internal {

// supported variants
// range(beg, end)
// range(C& cont)
// range(const T& x); // single item or drop this in favor of initializer list
// range(std::initializer_list<T>)
template <typename I, bool IS_INTEGER = false>
class IteratorRangeMaker {
  I m_beg;
  I m_end;

public:
  IteratorRangeMaker(const I& beg, const I& end) : m_beg(beg), m_end(end) {}

  template <typename Arg>
  auto operator()(const Arg&) const {
    return runtime::makeStandardRange(m_beg, m_end);
  }
};

template <typename I>
class IteratorRangeMaker<I, true> {
  I m_beg;
  I m_end;

public:
  IteratorRangeMaker(const I& beg, const I& end) : m_beg(beg), m_end(end) {}

  template <typename Arg>
  auto operator()(const Arg&) const {
    return runtime::makeStandardRange(boost::counting_iterator<I>(m_beg),
                                      boost::counting_iterator<I>(m_end));
  }
};

template <typename T>
class InitListRangeMaker {
  std::initializer_list<T> m_list;

public:
  explicit InitListRangeMaker(const std::initializer_list<T>& l) : m_list(l) {}

  template <typename Arg>
  auto operator()(const Arg&) const {
    return runtime::makeStandardRange(m_list.begin(), m_list.end());
  }
};

template <typename C, bool HAS_LOCAL_RANGE = true>
class ContainerRangeMaker {
  C& m_cont;

public:
  explicit ContainerRangeMaker(C& cont) : m_cont(cont) {}

  template <typename Arg>
  auto operator()(const Arg&) const {
    return runtime::makeLocalRange(m_cont);
  }
};

template <typename C>
class ContainerRangeMaker<C, false> {

  C& m_cont;

public:
  explicit ContainerRangeMaker(C& cont) : m_cont(cont) {}

  template <typename Arg>
  auto operator()(const Arg&) const {
    return runtime::makeStandardRange(m_cont.begin(), m_cont.end());
  }
};

template <typename C>
class HasLocalIter {

  template <typename T>
  using CallExprType = typename std::remove_reference<decltype(
      std::declval<T>().local_begin())>::type;

  template <typename T>
  static std::true_type go(typename std::add_pointer<CallExprType<T>>::type);

  template <typename T>
  static std::false_type go(...);

public:
  constexpr static const bool value =
      std::is_same<decltype(go<C>(nullptr)), std::true_type>::value;
};

} // end namespace internal

template <typename C>
auto iterate(C& cont) {
  return internal::ContainerRangeMaker<C, internal::HasLocalIter<C>::value>(
      cont);
}

template <typename T>
auto iterate(std::initializer_list<T> initList) {
  return internal::InitListRangeMaker<T>(initList);
}

template <typename I>
auto iterate(const I& beg, const I& end) {
  return internal::IteratorRangeMaker<I, std::is_integral<I>::value>(beg, end);
}

} // end namespace galois
#endif
