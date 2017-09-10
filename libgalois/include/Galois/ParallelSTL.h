/** Parallel STL equivalents -*- C++ -*-
 * @file
 *
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
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
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_PARALLELSTL_H
#define GALOIS_PARALLELSTL_H

#include "Galois/Accumulator.h"
#include "Galois/GaloisForwardDecl.h"
#include "Galois/NoDerefIterator.h"
#include "Galois/Traits.h"
#include "Galois/UserContext.h"
#include "Galois/WorkList/Chunked.h"

namespace Galois {
//! Parallel versions of STL library algorithms.
// TODO: rename to gstl?
namespace ParallelSTL {

template<typename Predicate, typename TO>
struct count_if_helper {
  Predicate f;
  GReducible<ptrdiff_t, TO>& ret;
  count_if_helper(Predicate p, GReducible<ptrdiff_t,TO>& c): f(p), ret(c) { }
  template<typename T>
  void operator()(const T& v) const {
    if (f(v)) ret.update(1);
  }
};

template<class InputIterator, class Predicate>
ptrdiff_t count_if(InputIterator first, InputIterator last, Predicate pred)
{
  auto R = [] (ptrdiff_t& lhs, ptrdiff_t rhs) { lhs += rhs; };
  GReducible<ptrdiff_t,decltype(R)> count(R);
  do_all(first, last, count_if_helper<Predicate, decltype(R)>(pred, count));
  return count.reduce();
}

template<typename InputIterator, class Predicate>
struct find_if_helper {
  typedef int tt_does_not_need_stats;
  typedef int tt_does_not_need_push;
  typedef int tt_does_not_need_aborts;
  typedef int tt_needs_parallel_break;

  typedef Galois::optional<InputIterator> ElementTy;
  typedef Substrate::PerThreadStorage<ElementTy> AccumulatorTy;
  AccumulatorTy& accum;
  Predicate& f;
  find_if_helper(AccumulatorTy& a, Predicate& p): accum(a), f(p) { }
  void operator()(const InputIterator& v, UserContext<InputIterator>& ctx) {
    if (f(*v)) {
      *accum.getLocal() = v;
      ctx.breakLoop();
    }
  }
};

template<class InputIterator, class Predicate>
InputIterator find_if(InputIterator first, InputIterator last, Predicate pred)
{
  typedef find_if_helper<InputIterator,Predicate> HelperTy;
  typedef typename HelperTy::AccumulatorTy AccumulatorTy;
  typedef Galois::WorkList::dChunkedFIFO<256> WL;
  AccumulatorTy accum;
  HelperTy helper(accum, pred);
  for_each(make_no_deref_iterator(first), make_no_deref_iterator(last), helper, Galois::wl<WL>());
  for (unsigned i = 0; i < accum.size(); ++i) {
    if (*accum.getRemote(i))
      return **accum.getRemote(i);
  }
  return last;
}

template<class Iterator>
Iterator choose_rand(Iterator first, Iterator last) {
  size_t dist = std::distance(first,last);
  if (dist)
    std::advance(first, rand() % dist);
  return first;
}

template<class Compare>
struct sort_helper {
  typedef int tt_does_not_need_aborts;
  Compare comp;
  
  //! Not equal in terms of less-than
  template<class value_type>
  struct neq_to: public std::binary_function<value_type,value_type,bool> {
    Compare comp;
    neq_to(Compare c): comp(c) { }
    bool operator()(const value_type& a, const value_type& b) const {
      return comp(a, b) || comp(b, a);
    }
  };

  sort_helper(Compare c): comp(c) { }

  template <class RandomAccessIterator, class Context>
  void operator()(std::pair<RandomAccessIterator,RandomAccessIterator> bounds, 
		  Context& ctx) {
    if (std::distance(bounds.first, bounds.second) <= 1024) {
      std::sort(bounds.first, bounds.second, comp);
    } else {
      typedef typename std::iterator_traits<RandomAccessIterator>::value_type VT;
      RandomAccessIterator pivot = choose_rand(bounds.first, bounds.second);
      VT pv = *pivot;
      pivot = std::partition(bounds.first, bounds.second,
          std::bind(comp, std::placeholders::_1, pv));
      //push the lower bit
      if (bounds.first != pivot)
	ctx.push(std::make_pair(bounds.first, pivot));
      //adjust the upper bit
      pivot = std::find_if(pivot, bounds.second, 
          std::bind(neq_to<VT>(comp), std::placeholders::_1, pv));
      //push the upper bit
      if (bounds.second != pivot)
	ctx.push(std::make_pair(pivot, bounds.second)); 
    }
  }
};

template<typename RandomAccessIterator, class Predicate>
std::pair<RandomAccessIterator, RandomAccessIterator>
dual_partition(RandomAccessIterator first1, RandomAccessIterator last1,
	       RandomAccessIterator first2, RandomAccessIterator last2,
	       Predicate pred) {
  typedef std::reverse_iterator<RandomAccessIterator> RI;
  RI first3(last2), last3(first2);
  while (true) {
    while (first1 != last1 && pred(*first1)) ++first1;
    if (first1 == last1) break;
    while (first3 != last3 && !pred(*first3)) ++first3;
    if (first3 == last3) break;
    std::swap(*first1++, *first3++);
  }
  return std::make_pair(first1, first3.base());
}

template<typename RandomAccessIterator, class Predicate>
struct partition_helper {
  typedef std::pair<RandomAccessIterator, RandomAccessIterator> RP;
  struct partition_helper_state {
    RandomAccessIterator first, last;
    RandomAccessIterator rfirst, rlast;
    Substrate::SimpleLock Lock;
    Predicate pred;
    typename std::iterator_traits<RandomAccessIterator>::difference_type BlockSize() { return 1024; }

    partition_helper_state(RandomAccessIterator f, RandomAccessIterator l, Predicate p)
      :first(f), last(l), rfirst(l), rlast(f), pred(p)
    {}
    RP takeHigh() {
      Lock.lock();
      unsigned BS = std::min(BlockSize(), std::distance(first,last));
      last -= BS;
      RandomAccessIterator rv = last;
      Lock.unlock();
      return std::make_pair(rv, rv+BS);
    }
    RP takeLow() {
      Lock.lock();
      unsigned BS = std::min(BlockSize(), std::distance(first,last));
      RandomAccessIterator rv = first;
      first += BS;
      Lock.unlock();
      return std::make_pair(rv, rv+BS);
    }
     void update(RP low, RP high) {
       Lock.lock();
       if (low.first != low.second) {
	 rfirst = std::min(rfirst, low.first);
	 rlast = std::max(rlast, low.second);
       }
       if (high.first != high.second) {
	 rfirst = std::min(rfirst, high.first);
	 rlast = std::max(rlast, high.second);
       }
       Lock.unlock();
     }
  };

  partition_helper(partition_helper_state* s) :state(s) {}

  partition_helper_state* state;

  void operator()(unsigned, unsigned) {
    RP high, low;
    do {
      RP parts = dual_partition(low.first, low.second, high.first, high.second, state->pred);
      low.first = parts.first;
      high.second = parts.second;
      if (low.first == low.second) low = state->takeLow();
      if (high.first == high.second) high = state->takeHigh();
    } while (low.first != low.second && high.first != high.second);
    state->update(low,high);
  }
};

template<class RandomAccessIterator, class Predicate>
RandomAccessIterator partition(RandomAccessIterator first, 
			       RandomAccessIterator last,
			       Predicate pred) {
  if (std::distance(first, last) <= 1024)
    return std::partition(first, last, pred);
  typedef partition_helper<RandomAccessIterator, Predicate> P;
  typename P::partition_helper_state s(first, last, pred);
  on_each(P(&s));
  if (s.rfirst == first && s.rlast == last) { //perfect !
    //abort();
    return s.first;
  }
  return std::partition(s.rfirst, s.rlast, pred);
}

struct pair_dist {
  template<typename RP>
  bool operator()(const RP& x, const RP& y) {
    return std::distance(x.first, x.second) > std::distance(y.first, y.second);
  }
};

template <class RandomAccessIterator,class Compare>
void sort(RandomAccessIterator first, RandomAccessIterator last, Compare comp) {
  if (std::distance(first, last) <= 1024) {
    std::sort(first, last, comp);
    return;
  }
  typedef Galois::WorkList::dChunkedFIFO<1> WL;
  typedef std::pair<RandomAccessIterator,RandomAccessIterator> Pair;
  Pair initial[1] = { std::make_pair(first, last) };
  
  for_each(&initial[0], &initial[1], sort_helper<Compare>(comp), Galois::wl<WL>());
}

template<class RandomAccessIterator>
void sort(RandomAccessIterator first, RandomAccessIterator last) {
  Galois::ParallelSTL::sort(first, last, std::less<typename std::iterator_traits<RandomAccessIterator>::value_type>());
}

template <class InputIterator, class T, typename BinaryOperation>
T accumulate (InputIterator first, InputIterator last, T init, const BinaryOperation& binary_op) {
  struct updater {
    BinaryOperation op;
    updater(const BinaryOperation& f) :op(f) {}
    void operator()(T& lhs, const T& rhs) { lhs = this->op(lhs, rhs); }
  };
  GReducible<T, updater> R{updater(binary_op)};
  R.update(init);
  do_all(first, last, [&R] (const T& v) { R.update(v); });
  return R.reduce(updater(binary_op));
}

template<class InputIterator, class T>
T accumulate(InputIterator first, InputIterator last, T init) {
  return accumulate(first, last, init, std::plus<T>());
}

template<typename T, typename MapFn, typename ReduceFn>
struct map_reduce_helper {
  Substrate::PerThreadStorage<T>& init;
  MapFn fn;
  ReduceFn reduce;
  map_reduce_helper(Galois::Substrate::PerThreadStorage<T>& i, MapFn fn, ReduceFn reduce)
    :init(i), fn(fn), reduce(reduce) {}
  template<typename U>
  void operator()(U&& v) const {
    *init.getLocal() = reduce(fn(std::forward<U>(v)), *init.getLocal());
  }
};

template<class InputIterator, class MapFn, class T, class ReduceFn>
T map_reduce(InputIterator first, InputIterator last, MapFn fn, T init, ReduceFn reduce) {
  Galois::Substrate::PerThreadStorage<T> reduced;
  do_all(first, last,
         map_reduce_helper<T,MapFn,ReduceFn>(reduced, fn, reduce));
  //         Galois::loopname("map_reduce"));
  for (unsigned i = 0; i < reduced.size(); ++i)
    init = reduce(init, *reduced.getRemote(i));
  return init;
}

}
}
#endif
