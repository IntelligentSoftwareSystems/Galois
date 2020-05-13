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

#ifndef GALOIS_PARALLELSTL_H
#define GALOIS_PARALLELSTL_H

#include "galois/config.h"
#include "galois/GaloisForwardDecl.h"
#include "galois/NoDerefIterator.h"
#include "galois/runtime/Range.h"
#include "galois/Reduction.h"
#include "galois/Traits.h"
#include "galois/UserContext.h"
#include "galois/worklists/Chunk.h"

namespace galois {
//! Parallel versions of STL library algorithms.
// TODO: rename to gstl?
namespace ParallelSTL {

template <class InputIterator, class Predicate>
size_t count_if(InputIterator first, InputIterator last, Predicate pred) {

  galois::GAccumulator<size_t> count;

  galois::do_all(galois::iterate(first, last), [&](const auto& v) {
    if (pred(v)) {
      count += 1;
    }
  });

  return count.reduce();
}

template <typename InputIterator, class Predicate>
struct find_if_helper {

  typedef galois::optional<InputIterator> ElementTy;
  typedef substrate::PerThreadStorage<ElementTy> AccumulatorTy;
  AccumulatorTy& accum;
  Predicate& f;

  find_if_helper(AccumulatorTy& a, Predicate& p) : accum(a), f(p) {}
  void operator()(const InputIterator& v, UserContext<InputIterator>& ctx) {
    if (f(*v)) {
      *accum.getLocal() = v;
      ctx.breakLoop();
    }
  }
};

template <class InputIterator, class Predicate>
InputIterator find_if(InputIterator first, InputIterator last, Predicate pred) {
  typedef find_if_helper<InputIterator, Predicate> HelperTy;
  typedef typename HelperTy::AccumulatorTy AccumulatorTy;
  typedef galois::worklists::PerSocketChunkFIFO<256> WL;
  AccumulatorTy accum;
  HelperTy helper(accum, pred);
  for_each(galois::iterate(make_no_deref_iterator(first),
                           make_no_deref_iterator(last)),
           helper, galois::no_conflicts(), galois::no_pushes(),
           galois::parallel_break(), galois::wl<WL>());
  for (unsigned i = 0; i < accum.size(); ++i) {
    if (*accum.getRemote(i))
      return **accum.getRemote(i);
  }
  return last;
}

template <class Iterator>
Iterator choose_rand(Iterator first, Iterator last) {
  size_t dist = std::distance(first, last);
  if (dist)
    std::advance(first, rand() % dist);
  return first;
}

template <class Compare>
struct sort_helper {
  Compare comp;

  //! Not equal in terms of less-than
  template <class value_type>
  struct neq_to : public std::binary_function<value_type, value_type, bool> {
    Compare comp;
    neq_to(Compare c) : comp(c) {}
    bool operator()(const value_type& a, const value_type& b) const {
      return comp(a, b) || comp(b, a);
    }
  };

  sort_helper(Compare c) : comp(c) {}

  template <class RandomAccessIterator, class Context>
  void operator()(std::pair<RandomAccessIterator, RandomAccessIterator> bounds,
                  Context& ctx) {
    if (std::distance(bounds.first, bounds.second) <= 1024) {
      std::sort(bounds.first, bounds.second, comp);
    } else {
      typedef
          typename std::iterator_traits<RandomAccessIterator>::value_type VT;
      RandomAccessIterator pivot = choose_rand(bounds.first, bounds.second);
      VT pv                      = *pivot;
      pivot                      = std::partition(bounds.first, bounds.second,
                             std::bind(comp, std::placeholders::_1, pv));
      // push the lower bit
      if (bounds.first != pivot)
        ctx.push(std::make_pair(bounds.first, pivot));
      // adjust the upper bit
      pivot =
          std::find_if(pivot, bounds.second,
                       std::bind(neq_to<VT>(comp), std::placeholders::_1, pv));
      // push the upper bit
      if (bounds.second != pivot)
        ctx.push(std::make_pair(pivot, bounds.second));
    }
  }
};

template <typename RandomAccessIterator, class Predicate>
std::pair<RandomAccessIterator, RandomAccessIterator>
dual_partition(RandomAccessIterator first1, RandomAccessIterator last1,
               RandomAccessIterator first2, RandomAccessIterator last2,
               Predicate pred) {
  typedef std::reverse_iterator<RandomAccessIterator> RI;
  RI first3(last2), last3(first2);
  while (true) {
    while (first1 != last1 && pred(*first1))
      ++first1;
    if (first1 == last1)
      break;
    while (first3 != last3 && !pred(*first3))
      ++first3;
    if (first3 == last3)
      break;
    std::swap(*first1++, *first3++);
  }
  return std::make_pair(first1, first3.base());
}

template <typename RandomAccessIterator, class Predicate>
struct partition_helper {
  typedef std::pair<RandomAccessIterator, RandomAccessIterator> RP;
  struct partition_helper_state {
    RandomAccessIterator first, last;
    RandomAccessIterator rfirst, rlast;
    substrate::SimpleLock Lock;
    Predicate pred;
    typename std::iterator_traits<RandomAccessIterator>::difference_type
    BlockSize() {
      return 1024;
    }

    partition_helper_state(RandomAccessIterator f, RandomAccessIterator l,
                           Predicate p)
        : first(f), last(l), rfirst(l), rlast(f), pred(p) {}
    RP takeHigh() {
      Lock.lock();
      unsigned BS = std::min(BlockSize(), std::distance(first, last));
      last -= BS;
      RandomAccessIterator rv = last;
      Lock.unlock();
      return std::make_pair(rv, rv + BS);
    }
    RP takeLow() {
      Lock.lock();
      unsigned BS = std::min(BlockSize(), std::distance(first, last));
      RandomAccessIterator rv = first;
      first += BS;
      Lock.unlock();
      return std::make_pair(rv, rv + BS);
    }
    void update(RP low, RP high) {
      Lock.lock();
      if (low.first != low.second) {
        rfirst = std::min(rfirst, low.first);
        rlast  = std::max(rlast, low.second);
      }
      if (high.first != high.second) {
        rfirst = std::min(rfirst, high.first);
        rlast  = std::max(rlast, high.second);
      }
      Lock.unlock();
    }
  };

  partition_helper(partition_helper_state* s) : state(s) {}

  partition_helper_state* state;

  void operator()(unsigned, unsigned) {
    RP high, low;
    do {
      RP parts  = dual_partition(low.first, low.second, high.first, high.second,
                                state->pred);
      low.first = parts.first;
      high.second = parts.second;
      if (low.first == low.second)
        low = state->takeLow();
      if (high.first == high.second)
        high = state->takeHigh();
    } while (low.first != low.second && high.first != high.second);
    state->update(low, high);
  }
};

template <class RandomAccessIterator, class Predicate>
RandomAccessIterator partition(RandomAccessIterator first,
                               RandomAccessIterator last, Predicate pred) {
  if (std::distance(first, last) <= 1024)
    return std::partition(first, last, pred);
  typedef partition_helper<RandomAccessIterator, Predicate> P;
  typename P::partition_helper_state s(first, last, pred);
  on_each(P(&s));
  if (s.rfirst == first && s.rlast == last) { // perfect !
    // abort();
    return s.first;
  }
  return std::partition(s.rfirst, s.rlast, pred);
}

struct pair_dist {
  template <typename RP>
  bool operator()(const RP& x, const RP& y) {
    return std::distance(x.first, x.second) > std::distance(y.first, y.second);
  }
};

template <class RandomAccessIterator, class Compare>
void sort(RandomAccessIterator first, RandomAccessIterator last, Compare comp) {
  if (std::distance(first, last) <= 1024) {
    std::sort(first, last, comp);
    return;
  }
  typedef galois::worklists::PerSocketChunkFIFO<1> WL;

  for_each(galois::iterate({std::make_pair(first, last)}),
           sort_helper<Compare>(comp), galois::no_conflicts(),
           galois::wl<WL>());
}

template <class RandomAccessIterator>
void sort(RandomAccessIterator first, RandomAccessIterator last) {
  galois::ParallelSTL::sort(
      first, last,
      std::less<
          typename std::iterator_traits<RandomAccessIterator>::value_type>());
}

template <class InputIterator, class T, typename BinaryOperation>
T accumulate(InputIterator first, InputIterator last, const T& identity,
             const BinaryOperation& binary_op) {

  auto id_fn = [=]() { return identity; };

  auto r = make_reducible(binary_op, id_fn);

  do_all(galois::iterate(first, last), [&](const T& v) { r.update(v); });

  return r.reduce();
}

template <class InputIterator, class T>
T accumulate(InputIterator first, InputIterator last, const T& identity = T()) {
  return accumulate(first, last, identity, std::plus<T>());
}

template <class InputIterator, class MapFn, class T, class ReduceFn>
T map_reduce(InputIterator first, InputIterator last, MapFn map_fn,
             ReduceFn reduce_fn, const T& identity) {

  auto id_fn = [=]() { return identity; };

  auto r = make_reducible(reduce_fn, id_fn);

  galois::do_all(galois::iterate(first, last),
                 [&](const auto& v) { r.update(map_fn(v)); });

  return r.reduce();
}

template <typename I>
std::enable_if_t<!std::is_scalar<internal::Val_ty<I>>::value> destroy(I first,
                                                                      I last) {
  using T = internal::Val_ty<I>;
  do_all(iterate(first, last), [=](T& i) { (&i)->~T(); });
}

template <class I>
std::enable_if_t<std::is_scalar<internal::Val_ty<I>>::value> destroy(I, I) {}

/**
 * Does a partial sum from first -> last and writes the results to the d_first
 * iterator.
 */
template <class InputIt, class OutputIt>
OutputIt partial_sum(InputIt first, InputIt last, OutputIt d_first) {
  using ValueType = typename std::iterator_traits<InputIt>::value_type;

  size_t sizeOfVector = std::distance(first, last);

  const size_t blockSize = 1 << 20;
  // optimization possible here: if num blocks = 1, then no need for 2 passes
  const size_t numBlocks = (sizeOfVector + blockSize - 1) / blockSize;

  std::vector<ValueType> localSums(numBlocks);

  // get the block sums
  galois::do_all(
      galois::iterate((size_t)0, numBlocks), [&](const size_t& block) {
        ValueType lsum = 0;
        // loop over block
        size_t blockEnd = std::min((block + 1) * blockSize, sizeOfVector);
        for (size_t i = block * blockSize; i < blockEnd; i++) {
          lsum += *(first + i);
        }
        localSums[block] = lsum;
      });

  // bulkPrefix[i] holds the starting sum of a particular block i
  std::vector<ValueType> bulkPrefix(numBlocks);
  // first block starts at 0; rest start at sum of all previous blocks
  ValueType total = 0;
  for (size_t block = 0; block < numBlocks; block++) {
    bulkPrefix[block] = total;
    total += localSums[block];
  }

  galois::do_all(
      galois::iterate((size_t)0, numBlocks), [&](const size_t& block) {
        // start with this block's prefix up to this point
        ValueType localTotal = bulkPrefix[block];
        size_t blockEnd      = std::min((block + 1) * blockSize, sizeOfVector);

        for (size_t i = block * blockSize; i < blockEnd; i++) {
          // add current element first, then save
          localTotal += *(first + i);
          *(d_first + i) = localTotal;
        }
      });

  // return the iterator past the last element written
  return d_first + sizeOfVector;
}

} // end namespace ParallelSTL
} // end namespace galois
#endif
