/** Parallel STL equivalents -*- C++ -*-
 * @file
 *
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
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_PARALLELSTL_H
#define GALOIS_PARALLELSTL_H

#include "galois/Reduction.h"
#include "galois/GaloisForwardDecl.h"
#include "galois/NoDerefIterator.h"
#include "galois/Traits.h"
#include "galois/UserContext.h"
#include "galois/worklists/Chunked.h"
#include "galois/runtime/Range.h"

namespace galois {
//! Parallel versions of STL library algorithms.
// TODO: rename to gstl?
namespace ParallelSTL {


template<class InputIterator, class Predicate>
size_t count_if(InputIterator first, InputIterator last, Predicate pred)
{

  galois::GAccumulator<size_t> count;

  galois::do_all(galois::iterate(first, last), 
      [&] (const auto& v) {
        if (pred(v)) {
          count += 1;
        }
      });

  return count.reduce();
}

template<typename InputIterator, class Predicate>
struct find_if_helper {

  typedef galois::optional<InputIterator> ElementTy;
  typedef substrate::PerThreadStorage<ElementTy> AccumulatorTy;
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
  typedef galois::worklists::dChunkedFIFO<256> WL;
  AccumulatorTy accum;
  HelperTy helper(accum, pred);
  for_each(galois::iterate(make_no_deref_iterator(first), make_no_deref_iterator(last)), helper,
      galois::no_conflicts(),
      galois::no_pushes(),
      galois::parallel_break(),
      galois::wl<WL>());
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
    substrate::SimpleLock Lock;
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
  typedef galois::worklists::dChunkedFIFO<1> WL;

  for_each(galois::iterate( { std::make_pair(first, last) } )
      , sort_helper<Compare>(comp)
      , galois::no_conflicts()
      , galois::wl<WL>());
}

template<class RandomAccessIterator>
void sort(RandomAccessIterator first, RandomAccessIterator last) {
  galois::ParallelSTL::sort(first, last, std::less<typename std::iterator_traits<RandomAccessIterator>::value_type>());
}

template <class InputIterator, class T, typename BinaryOperation>
T accumulate (InputIterator first, InputIterator last, const T& identity, const BinaryOperation& binary_op) {

  GSimpleReducible<BinaryOperation, T> R(binary_op, identity);

  do_all(galois::iterate(first, last), [&R] (const T& v) { R.update(v); });
  return R.reduce();
}

template<class InputIterator, class T>
T accumulate(InputIterator first, InputIterator last, const T& identity=T()) {
  return accumulate(first, last, identity, std::plus<T>());
}
template<class InputIterator, class MapFn, class T, class ReduceFn>
T map_reduce(InputIterator first, InputIterator last, MapFn mapFn, ReduceFn reduceFn, const T& identity) {

  galois::GSimpleReducible<ReduceFn, T> reducer(reduceFn, identity);

  galois::do_all(galois::iterate(first, last),
      [&] (const auto& v) {
        reducer.update(mapFn(v));
      });

  return reducer.reduce();

}

template<typename I>
std::enable_if_t<!std::is_scalar<internal::Val_ty<I> >::value> destroy (I first, I last) {
  using T = internal::Val_ty<I>;
  do_all(iterate(first, last),
    [=] (T& i) {
      (&i)->~T();
    });
    
}

template<class I>
std::enable_if_t<std::is_scalar<internal::Val_ty<I> >::value> destroy (I, I) { }

} // end namespace ParallelSTL
} // end namespace galois
#endif
