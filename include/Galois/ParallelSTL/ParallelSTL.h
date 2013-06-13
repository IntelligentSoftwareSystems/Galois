/** Parallel STL equivalents -*- C++ -*-
 * @file
 * This is the only file to include for basic Galois functionality.
 *
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
#ifndef GALOIS_PARALLELSTL_PARALLELSTL_H
#define GALOIS_PARALLELSTL_PARALLELSTL_H

#include "Galois/UserContext.h"
#include "Galois/NoDerefIterator.h"
#include "Galois/WorkList/WorkList.h"
#include "Galois/Runtime/ParallelWork.h"
#include "Galois/Runtime/DoAll.h"

namespace Galois {
//! Parallel versions of STL library algorithms.
namespace ParallelSTL {

template<typename Predicate>
struct count_if_helper {
  Predicate f;
  ptrdiff_t ret;
  count_if_helper(Predicate p): f(p), ret(0) { }
  template<typename T>
  void operator()(const T& v) {
    if (f(v)) ++ret;
  }
};

struct count_if_reducer {
  template<typename CIH>
  void operator()(CIH& dest, const CIH& src) {
    dest.ret += src.ret;
  }
};

template<class InputIterator, class Predicate>
ptrdiff_t count_if(InputIterator first, InputIterator last, Predicate pred)
{
  return Runtime::do_all_impl(Runtime::makeStandardRange(first, last),
			      count_if_helper<Predicate>(pred), count_if_reducer()).ret;
}

template<typename InputIterator, class Predicate>
struct find_if_helper {
  typedef int tt_does_not_need_stats;
  typedef int tt_does_not_need_parallel_push;
  typedef int tt_does_not_need_aborts;
  typedef int tt_needs_parallel_break;

  typedef Galois::optional<InputIterator> ElementTy;
  typedef Runtime::PerThreadStorage<ElementTy> AccumulatorTy;
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
  Runtime::for_each_impl<WL>(Runtime::makeStandardRange(
        make_no_deref_iterator(first),
        make_no_deref_iterator(last)), helper, 0);
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
		  Context& cnx) {
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
	cnx.push(std::make_pair(bounds.first, pivot));
      //adjust the upper bit
      pivot = std::find_if(pivot, bounds.second, 
          std::bind(neq_to<VT>(comp), std::placeholders::_1, pv));
      //push the upper bit
      if (bounds.second != pivot)
	cnx.push(std::make_pair(pivot, bounds.second)); 
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
    Runtime::LL::SimpleLock<true> Lock;
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
  Runtime::on_each_impl(P(&s), 0);
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
  
  Runtime::for_each_impl<WL>(Runtime::makeStandardRange(&initial[0], &initial[1]), sort_helper<Compare>(comp), 0);
}

template<class RandomAccessIterator>
void sort(RandomAccessIterator first, RandomAccessIterator last) {
  Galois::ParallelSTL::sort(first, last, std::less<typename std::iterator_traits<RandomAccessIterator>::value_type>());
}

template<typename T, typename BinOp>
struct accumulate_helper {
  T init;
  BinOp op;
  accumulate_helper(T i, BinOp o) :init(i), op(o) {}
  void operator()(const T& v) {
    init = op(init,v);
  }
};

template<typename BinOp>
struct accumulate_helper_reduce {
  BinOp op;
  accumulate_helper_reduce(BinOp o) :op(o) {}
  template<typename T>
  void operator()(T& dest, const T& src) const {
    dest.init = op(dest.init, src.init);
  }
};

template <class InputIterator, class T, typename BinaryOperation>
T accumulate (InputIterator first, InputIterator last, T init, BinaryOperation binary_op) {
  return Runtime::do_all_impl(Runtime::makeStandardRange(first, last),
      accumulate_helper<T,BinaryOperation>(init, binary_op),
      accumulate_helper_reduce<BinaryOperation>(binary_op)).init;
}

template<class InputIterator, class T>
T accumulate(InputIterator first, InputIterator last, T init) {
  return accumulate(first, last, init, std::plus<T>());
}

template<typename T, typename MapFn, typename ReduceFn>
struct map_reduce_helper {
  T init;
  MapFn fn;
  ReduceFn reduce;
  map_reduce_helper(T i, MapFn fn, ReduceFn reduce) :init(i), fn(fn), reduce(reduce) {}
  template<typename U>
  void operator()(U&& v) {
    init = reduce(fn(std::forward<U>(v)), init);
  }
};

template<class InputIterator, class MapFn, class T, class ReduceFn>
T map_reduce(InputIterator first, InputIterator last, MapFn fn, T init, ReduceFn reduce) {
  return Runtime::do_all_impl(Runtime::makeStandardRange(first, last),
      map_reduce_helper<T,MapFn,ReduceFn>(init, fn, reduce),
      accumulate_helper_reduce<ReduceFn>(reduce)).init;
}

}
}
#endif
