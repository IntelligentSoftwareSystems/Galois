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

#ifndef GALOIS_REDUCTION_H
#define GALOIS_REDUCTION_H

#include "galois/gstl.h"
#include "galois/substrate/PerThreadStorage.h"

#include <limits>

namespace galois {

/**
 * GSimpleReducible stores per thread values of a variable of type T, suitable
 * for small, cheap to copy, plain T types, where T is not a container, is small
 * and does not allocate memory
 *
 * The final value is obtained by performing a reduction on per thread values
 * using the provided binary functor BinFunc. BinFunc conforms to STL semantics,
 * such as std::plus<T>, with signature:
 *  T operator()(const T& lhs, const T& rhs)
 *
 * The identity value provided to the constructor is the identity element for
 * the binary functor. i.e. f(x, identity) == x
 */
template <typename BinFunc, typename T>
class GSimpleReducible {

protected:
  BinFunc m_func;
  const T m_identity;
  galois::substrate::PerThreadStorage<T> m_data;

  void initialize(void) {
    for (unsigned i = 0; i < m_data.size(); ++i) {
      *(m_data.getRemote(i)) = m_identity;
    }
  }

public:
  using AccumType = T; // for outside access if necessary

  /**
   * @param func the binary functor acting as the reduction operator
   * @param identity is the identity value for the functor f,
   * i.e., f(x,identity) == x
   */
  explicit GSimpleReducible(const BinFunc& func = BinFunc(),
                            const T& identity   = T())
      : m_func(func), m_identity(identity) {
    initialize();
  }

  /**
   * Updates the thread local value by applying the reduction operator to
   * current and newly provided value
   */
  template <typename T2>
  void update(const T2& rhs) {
    T& lhs = *m_data.getLocal();
    lhs    = m_func(lhs, rhs);
  }

  /**
   * Returns the final reduction value. Only valid outside the parallel region.
   */
  T reduce() const {
    T res = *m_data.getLocal();
    for (unsigned int i = 1; i < m_data.size(); ++i) {
      const T& d = *m_data.getRemote(i);
      res        = m_func(res, d);
    }
    return res;
  }

  /**
   * Returns the final reduction value. Only valid outside the parallel region.
   */
  template <typename FnAlt>
  T reduce(FnAlt fn) {
    T res = *m_data.getLocal();
    for (unsigned int i = 1; i < m_data.size(); ++i) {
      const T& d = *m_data.getRemote(i);
      res        = fn(res, d);
    }
    return res;
  }

  /**
   * reset value
   */
  void reset() {
    for (unsigned int i = 0; i < m_data.size(); ++i) {
      *m_data.getRemote(i) = m_identity;
    }
  }

  /**
   * @return read the current local value for this thread
   * performs an unsynchronized read
   *
   */
  T peekLocal(void) const { return *m_data.getLocal(); }

  /**
   * @param tid thread id
   * reads (in unsynchronized manner) the value of another thread 'tid'
   * value may be incorrect and may be usafe to use as it may return an object
   * in an intermediate state while its being updated
   */
  T peekRemote(unsigned tid) { return *m_data.getRemote(tid); }

  size_t size(void) const { return m_data.size(); }
};

//! Operator form of max
template <typename T>
struct gmax {
  const T& operator()(const T& lhs, const T& rhs) const {
    return std::max<T>(lhs, rhs);
  }
};

//! Operator form of min
template <typename T>
struct gmin {
  const T& operator()(const T& lhs, const T& rhs) const {
    return std::min<T>(lhs, rhs);
  }
};

//! Accumulator for T where accumulation is sum
template <typename T>
class GAccumulator : public GSimpleReducible<std::plus<T>, T> {
  typedef GSimpleReducible<std::plus<T>, T> base_type;

public:
  GAccumulator& operator+=(const T& rhs) {
    base_type::update(rhs);
    return *this;
  }

  GAccumulator& operator-=(const T& rhs) {
    base_type::update(-rhs);
    return *this;
  }
};

//! Accumulator for T where accumulation is max
template <typename T>
class GReduceMax : public GSimpleReducible<gmax<T>, T> {
  typedef GSimpleReducible<gmax<T>, T> base_type;

public:
  GReduceMax() : base_type(gmax<T>(), std::numeric_limits<T>::min()) {}
};

//! Accumulator for T where accumulation is min
template <typename T>
class GReduceMin : public GSimpleReducible<gmin<T>, T> {
  typedef GSimpleReducible<gmin<T>, T> base_type;

public:
  GReduceMin() : base_type(gmin<T>(), std::numeric_limits<T>::max()) {}
};

//! logical AND reduction
class GReduceLogicalAND
    : public GSimpleReducible<std::logical_and<bool>, bool> {
  typedef GSimpleReducible<std::logical_and<bool>, bool> base_type;

public:
  GReduceLogicalAND(void) : base_type(std::logical_and<bool>(), true) {}
};

class GReduceLogicalOR : public GSimpleReducible<std::logical_or<bool>, bool> {
  typedef GSimpleReducible<std::logical_or<bool>, bool> base_type;

public:
  GReduceLogicalOR(void) : base_type(std::logical_or<bool>(), false) {}
};

/**
 * GBigReducible stores per thread values of a variable of type T. Suitable
 * for large objects, objects that are not trivially copyable or are inefficient
 * to copy, such as containers, or objects that allocate memory internally
 *
 * The final value is obtained by performing a reduction on per thread values
 * and accumulating the result on master thread (thread 0),
 * using the provided binary functor BinFunc. BinFunc updates values in place
 * and conforms to:
 *
 *  void operator()(T& lhs, const T& rhs)
 *
 * The identity value provided to the constructor is the identity element for
 * the i binary functor. i.e. f(x, identity) == x
 */
template <typename BinFunc, typename T>
class GBigReducible {
protected:
  BinFunc m_func;
  galois::substrate::PerThreadStorage<T> m_data;
  const T m_identity;

  void initialize(void) {
    for (unsigned i = 0; i < m_data.size(); ++i) {
      *(m_data.getRemote(i)) = m_identity;
    }
  }

public:
  /**
   * @param f the binary functor acting as the reduction operator
   * @param identity is the identity value for the functor f,
   * i.e., f(x,identity) == x
   */
  explicit GBigReducible(const BinFunc& f = BinFunc(), const T& identity = T())
      : m_func(f), m_identity(identity) {
    initialize();
  }

  /**
   * Updates the thread local value by applying the reduction operator to
   * current and newly provided value
   */
  template <typename T2>
  void update(const T2& rhs) {
    T& lhs = *m_data.getLocal();
    m_func(lhs, rhs);
  }

  /**
   * Returns the final reduction value. Only valid outside the parallel region.
   */
  T& reduce() {
    T& d0 = *m_data.getLocal();
    for (unsigned int i = 1; i < m_data.size(); ++i) {
      T& d = *m_data.getRemote(i);
      m_func(d0, d);
      d = m_identity;
    }
    return d0;
  }

  /**
   * Returns the final reduction value. Only valid outside the parallel region.
   */
  template <typename FnAlt>
  T& reduce(FnAlt fn) {
    T& d0 = *m_data.getLocal();
    for (unsigned int i = 1; i < m_data.size(); ++i) {
      T& d = *m_data.getRemote(i);
      fn(d0, d);
      d = m_identity;
    }
    return d0;
  }

  /**
   * reset value
   */
  void reset() {
    for (unsigned int i = 0; i < m_data.size(); ++i) {
      *m_data.getRemote(i) = m_identity;
    }
  }

  T& peekLocal(void) { return *m_data.getLocal(); }

  const T& peekLocal(void) const { return *m_data.getLocal(); }
};

namespace internal {

//! Turns binary functions over values into functions over references
//!
//! T operator()(const T& a, const T& b) =>
//! void operator()(T& a, const T& b)
template <typename BinFunc>
struct ReduceAssignWrap {
  BinFunc fn;
  ReduceAssignWrap(const BinFunc& f = BinFunc()) : fn(f) {}
  template <typename T>
  void operator()(T& lhs, const T& rhs) const {
    lhs = fn(lhs, rhs);
  }
};

//! Turns binary functions over item references into functions over vectors of
//! items
//!
//! void operator()(T& a, const T& b) =>
//! void operator()(std::vector<T>& a, const std::vector<T>& b)
template <typename BinFunc, typename T>
struct ReduceVecPerItemWrap {

  BinFunc func;
  const T elemIdentity;

  explicit ReduceVecPerItemWrap(const BinFunc& f  = BinFunc(),
                                const T& identity = T())
      : func(f), elemIdentity(identity) {}

  template <typename C>
  void operator()(C& lhs, const C& rhs) const {
    if (lhs.size() < rhs.size())
      lhs.resize(rhs.size(), elemIdentity);
    typename C::iterator ii = lhs.begin();
    for (typename C::const_iterator jj = rhs.begin(), ej = rhs.end(); jj != ej;
         ++ii, ++jj) {
      func(*ii, *jj);
    }
  }
};

//! Turns binary functions over item (value) references into functions over maps
//! of items
//!
//! void operator()(V& a, const V& b) =>
//! void operator()(std::map<K,V>& a, const std::map<K,V>& b)
template <typename BinFunc, typename T>
struct ReduceMapPerItemWrap {
  BinFunc func;
  const T elemIdentity;

  ReduceMapPerItemWrap(const BinFunc& f = BinFunc(), const T& identity = T())
      : func(f), elemIdentity(identity) {}

  template <typename C>
  void operator()(C& lhs, const C& rhs) const {
    for (typename C::const_iterator jj = rhs.begin(), ej = rhs.end(); jj != ej;
         ++jj) {
      lhs.insert(typename C::value_type(
          jj->first, elemIdentity)); // rhs key must be in lhs for lhs[] to work
      func(lhs[jj->first], jj->second);
    }
  }
};

template <typename ItemReduceFn, typename T>
using VecItemReduceFn = ReduceVecPerItemWrap<ReduceAssignWrap<ItemReduceFn>, T>;

template <typename ItemReduceFn, typename T>
using MapItemReduceFn = ReduceMapPerItemWrap<ReduceAssignWrap<ItemReduceFn>, T>;
} // end namespace internal

//! Accumulator for vector where a vector is treated as a map and accumulate
//! does element-wise addition among all entries
template <typename T, typename ItemReduceFn = std::plus<T>,
          typename VecTy = galois::gstl::Vector<T>>
class GVectorPerItemReduce
    : public GBigReducible<internal::VecItemReduceFn<ItemReduceFn, T>, VecTy> {

  using VecReduceFn = internal::VecItemReduceFn<ItemReduceFn, T>;
  using base_type   = GBigReducible<VecReduceFn, VecTy>;

  static_assert(std::is_same<typename VecTy::value_type, T>::value,
                "T doesn't match VecTy::value_type");

  VecReduceFn vecFn;

public:
  using container_type = VecTy;
  using value_type     = typename VecTy::value_type;

  GVectorPerItemReduce(const ItemReduceFn& func = ItemReduceFn(),
                       const T& identity        = T())
      : base_type(), vecFn(func, identity) {}

  void resizeAll(size_t s) {
    for (int i = 0; i < this->m_data.size(); ++i) {
      this->m_data.getRemote(i)->resize(s, vecFn.elemIdentity);
    }
  }

  void update(size_t index, const T& value) {
    VecTy& v = *this->m_data.getLocal();
    if (v.size() <= index) {
      v.resize(index + 1, vecFn.elemIdentity);
    }
    vecFn.func(v[index], value);
  }
};

template <typename T, typename ItemReduceFn = std::plus<T>,
          typename Deq = galois::gstl::Deque<T>>
using GDequePerItemReduce = GVectorPerItemReduce<T, ItemReduceFn, Deq>;

//! Accumulator for map where accumulate does element-wise addition among
//! all entries
template <typename K, typename V, typename ItemReduceFn = std::plus<V>,
          typename C = std::less<K>, typename MapTy = gstl::Map<K, V, C>>
class GMapPerItemReduce
    : public GBigReducible<internal::MapItemReduceFn<ItemReduceFn, V>, MapTy> {

  using MapReduceFn = internal::MapItemReduceFn<ItemReduceFn, V>;
  using base_type   = GBigReducible<MapReduceFn, MapTy>;

  static_assert(std::is_same<typename MapTy::key_type, K>::value,
                "K doesn't match MapTy::key_type");
  static_assert(std::is_same<typename MapTy::mapped_type, V>::value,
                "V doesn't match MapTy::mapped_type");

  MapReduceFn mapFn;

public:
  using container_type = MapTy;
  using value_type     = typename MapTy::value_type;

  GMapPerItemReduce(const ItemReduceFn& func = ItemReduceFn(),
                    const V& identity        = V())
      : base_type(), mapFn(func, identity) {}

  void update(const K& key, const V& value) {
    MapTy& v = *this->m_data.getLocal();
    v.insert(typename MapTy::value_type(
        key,
        mapFn.elemIdentity)); // insert v[key] if absent, v[] must return valid
    mapFn.func(v[key], value);
  }
};

//! Turns functions over elements of a range into functions over collections
//!
//! void operator()(T a) =>
//! void operator()(Collection<T>& a, const Collection<T>& b)
// template<typename CollectionTy,template<typename> class AdaptorTy>
// struct ReduceCollectionWrap {
// typedef typename CollectionTy::value_type value_type;
//
// void operator()(CollectionTy& lhs, const CollectionTy& rhs) {
// AdaptorTy<CollectionTy> adapt(lhs, lhs.begin());
// std::copy(rhs.begin(), rhs.end(), adapt);
// }
//
// void operator()(CollectionTy& lhs, const value_type& rhs) {
// AdaptorTy<CollectionTy> adapt(lhs, lhs.begin());
// *adapt = rhs;
// }
// };
//

//! General accumulator for collections following STL interface where
//! accumulate means collection union. Since union/append/push_back are
//! not standard among collections, the AdaptorTy template parameter
//! allows users to provide an iterator adaptor along the lines of
// //! std::inserter or std::back_inserter.
// template<typename CollectionTy,template<typename> class AdaptorTy>
// class GCollectionAccumulator: public GBigReducible<CollectionTy,
// ReduceCollectionWrap<CollectionTy, AdaptorTy> > { typedef
// ReduceCollectionWrap<CollectionTy, AdaptorTy> Func; typedef
// GBigReducible<CollectionTy, Func> base_type; typedef typename
// CollectionTy::value_type value_type;
//
// Func func;
//
// public:
// void update(const value_type& rhs) {
// CollectionTy& v = *this->m_data.getLocal();
// func(v, rhs);
// }
// };

namespace internal {

template <typename C>
struct VecMerger : public std::binary_function<void, C&, const C&> {

  void operator()(C& lhs, const C& rhs) const {
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
  }

  void operator()(C& lhs, const typename C::value_type& val) const {
    lhs.push_back(val);
  }
};

template <typename C>
struct SetMerger : public std::binary_function<void, C&, const C&> {

  void operator()(C& lhs, const C& rhs) const {
    lhs.insert(rhs.begin(), rhs.end());
  }

  void operator()(C& lhs, const typename C::value_type& val) const {
    lhs.insert(val);
  }
};

template <typename C>
struct ListMerger : public std::binary_function<void, C&, const C&> {

  void operator()(C& lhs, const C& rhs) const {
    lhs.splice(lhs.end(), static_cast<C&>(rhs));
  }

  void operator()(C& lhs, const typename C::value_type& val) const {
    lhs.push_front(val);
  }
};

template <typename MergeFunc, typename C>
struct ContAccum : public GBigReducible<MergeFunc, C> {

  using Base = GBigReducible<MergeFunc, C>;

  using container_type = C;
  using value_type     = typename C::value_type;

  ContAccum(const MergeFunc& f = MergeFunc(), const C& identity = C())
      : Base(f, identity) {}

  void update(const typename C::value_type& val) {
    auto& myCont = *Base::m_data.getLocal();
    Base::m_func(myCont, val);
  }
};

} // end namespace internal

template <typename T, typename VecTy = gstl::Vector<T>>
class GVectorAccumulator
    : public internal::ContAccum<internal::VecMerger<VecTy>, VecTy> {};

template <typename T, typename Deq = gstl::Deque<T>>
class GDequeAccumulator
    : public internal::ContAccum<internal::VecMerger<Deq>, Deq> {};

template <typename T, typename C = std::less<T>,
          typename SetTy = gstl::Set<T, C>>
class GSetAccumulator
    : public internal::ContAccum<internal::SetMerger<SetTy>, SetTy> {};

template <typename T, typename ListTy = gstl::List<T>>
class GListAccumulator
    : public internal::ContAccum<internal::ListMerger<ListTy>, ListTy> {};
//! Accumulator for set where accumulation is union
// template<typename SetTy>
// class GSetAccumulator: public GCollectionAccumulator<SetTy,
// std::insert_iterator> { };
//
// //! Accumulator for vector where accumulation is concatenation
// template<typename VecTy>
// class GVectorAccumulator: public GCollectionAccumulator<VecTy,
// std::back_insert_iterator> { };

} // namespace galois
#endif // GALOIS_REDUCTION_H
