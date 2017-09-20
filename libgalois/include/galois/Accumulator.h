/** Accumulator type -*- C++ -*-
 * @file
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

#ifndef GALOIS_ACCUMULATOR_H
#define GALOIS_ACCUMULATOR_H

#include "galois/substrate/PerThreadStorage.h"

#include <limits>

namespace galois {

/**
 * GReducible stores per thread values of a variable of type T
 *
 * The final value is obtained by performing a reduction on per thread values
 * using the provided binary functor BinFunc. BinFunc updates values in place
 * and conforms to:
 *
 *  void operator()(T& lhs, const T& rhs)
 *
 * The identity value provided to the constructor is the identity element for the i
 * binary functor.
 * i.e. f(x, identity) == x
 */
template<typename T, typename BinFunc>
class GReducible {
protected:
  BinFunc m_func;
  galois::substrate::PerThreadStorage<T> m_data;
  const T m_identity;

  void initialize (void) {
    for (unsigned i = 0; i < m_data.size (); ++i) {
      *(m_data.getRemote (i)) = m_identity;
    }
  }

public:

  /**
   * @param f the binary functor acting as the reduction operator
   * @param identity is the identity value for the functor f, 
   * i.e., f(x,identity) == x
   */
  explicit GReducible(const BinFunc& f, const T& identity): m_func(f), m_identity(identity) { 
    initialize ();
  }

  /**
   * @param f the binary functor acting as the reduction operator
   * default constructor T() assumed to yield identity value
   */
  explicit GReducible(const BinFunc& f = BinFunc()): m_func(f), m_identity(T()) {
    initialize ();
  }

  /**
   * Updates the thread local value by applying the reduction operator to
   * current and newly provided value
   */
  template<typename T2>
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
  template<typename FnAlt>
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
};


//! Operator form of max
template<typename T>
struct gmax {
  const T& operator()(const T& lhs, const T& rhs) const {
    return std::max<T>(lhs, rhs);
  }
};

//! Operator form of min
template<typename T>
struct gmin {
  const T& operator()(const T& lhs, const T& rhs) const {
    return std::min<T>(lhs, rhs);
  }
};

//! Turns binary functions over values into functions over references
//!
//! T operator()(const T& a, const T& b) =>
//! void operator()(T& a, const T& b)
template<typename BinFunc>
struct ReduceAssignWrap {
  BinFunc fn;
  ReduceAssignWrap(const BinFunc& f = BinFunc()): fn(f) { }
  template<typename T>
  void operator()(T& lhs, const T& rhs) const {
    lhs = fn(lhs, rhs);
  }
};

//! Turns binary functions over item references into functions over vectors of items
//!
//! void operator()(T& a, const T& b) =>
//! void operator()(std::vector<T>& a, const std::vector<T>& b)
template<typename BinFunc>
struct ReduceVectorWrap {
  BinFunc fn;
  ReduceVectorWrap(const BinFunc& f = BinFunc()): fn(f) { }
  template<typename T>
  void operator()(T& lhs, const T& rhs) const {
    if (lhs.size() < rhs.size())
      lhs.resize(rhs.size());
    typename T::iterator ii = lhs.begin();
    for (typename T::const_iterator jj = rhs.begin(), ej = rhs.end(); jj != ej; ++ii, ++jj) {
      fn(*ii, *jj);
    }
  }
};

//! Turns binary functions over item (value) references into functions over maps of items
//!
//! void operator()(V& a, const V& b) =>
//! void operator()(std::map<K,V>& a, const std::map<K,V>& b)
template<typename BinFunc>
struct ReduceMapWrap {
  BinFunc fn;
  ReduceMapWrap(const BinFunc& f = BinFunc()): fn(f) { }
  template<typename T>
  void operator()(T& lhs, const T& rhs) const {
    for (typename T::const_iterator jj = rhs.begin(), ej = rhs.end(); jj != ej; ++jj) {
      fn(lhs[jj->first], jj->second);
    }
  }
};

//! Turns functions over elements of a range into functions over collections
//!
//! void operator()(T a) =>
//! void operator()(Collection<T>& a, const Collection<T>& b)
template<typename CollectionTy,template<class> class AdaptorTy>
struct ReduceCollectionWrap {
  typedef typename CollectionTy::value_type value_type;

  void operator()(CollectionTy& lhs, const CollectionTy& rhs) {
    AdaptorTy<CollectionTy> adapt(lhs, lhs.begin());
    std::copy(rhs.begin(), rhs.end(), adapt);
  }

  void operator()(CollectionTy& lhs, const value_type& rhs) {
    AdaptorTy<CollectionTy> adapt(lhs, lhs.begin());
    *adapt = rhs;
  }
};

/**
 * Simplification of GReducible where BinFunc calculates results by
 * value, i.e., BinFunc conforms to:
 *
 *  T operator()(const T& a, const T& b);
 */
template<typename T, typename BinFunc>
class GSimpleReducible: public GReducible<T, ReduceAssignWrap<BinFunc> >  {
  typedef GReducible<T, ReduceAssignWrap<BinFunc> > base_type;
public:
  explicit GSimpleReducible(const BinFunc& func = BinFunc(), const T& identity=T()): base_type(func, identity) { }

  //! read-only reduction on values computed by each thread, 
  //! valid outside parallel section
  //! use inside parallel section at your own risk
  T reduceRO () const {
    T d0 = *this->m_data.getRemote(0);
    for (unsigned int i = 1; i < this->m_data.size(); ++i) {
      const T& d = *this->m_data.getRemote(i);
      this->m_func(d0, d);
    }
    return d0;
  }
};

//! Accumulator for T where accumulation is sum
template<typename T>
class GAccumulator: public GSimpleReducible<T, std::plus<T> > {
  typedef GSimpleReducible<T, std::plus<T> > base_type;

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

//! General accumulator for collections following STL interface where
//! accumulate means collection union. Since union/append/push_back are
//! not standard among collections, the AdaptorTy template parameter
//! allows users to provide an iterator adaptor along the lines of
//! std::inserter or std::back_inserter.
template<typename CollectionTy,template<class> class AdaptorTy>
class GCollectionAccumulator: public GReducible<CollectionTy, ReduceCollectionWrap<CollectionTy, AdaptorTy> > {
  typedef ReduceCollectionWrap<CollectionTy, AdaptorTy> Func;
  typedef GReducible<CollectionTy, Func> base_type;
  typedef typename CollectionTy::value_type value_type;

  Func func;

public:
  void update(const value_type& rhs) {
    CollectionTy& v = *this->m_data.getLocal();
    func(v, rhs);
  }
};

//! Accumulator for set where accumulation is union
template<typename SetTy>
class GSetAccumulator: public GCollectionAccumulator<SetTy, std::insert_iterator> { };

//! Accumulator for vector where accumulation is concatenation
template<typename VectorTy>
class GVectorAccumulator: public GCollectionAccumulator<VectorTy, std::back_insert_iterator> { };

//! Accumulator for vector where a vector is treated as a map and accumulate
//! does element-wise addition among all entries
template<typename VectorTy>
class GVectorElementAccumulator: public GReducible<VectorTy, ReduceVectorWrap<ReduceAssignWrap<std::plus<typename VectorTy::value_type> > > > {
  typedef ReduceAssignWrap<std::plus<typename VectorTy::value_type> > ElementFunc;
  typedef GReducible<VectorTy, ReduceVectorWrap<ElementFunc> > base_type;
  typedef typename VectorTy::value_type value_type;

  ElementFunc func;

public:

  void resize(size_t s) {
    for (int i = 0; i < this->m_data.size(); ++i)
      this->m_data.getRemote(i)->resize(s);
  }

  VectorTy& getLocal() {
    return *this->m_data.getLocal();
  }

  void update(size_t index, const value_type& rhs) {
    VectorTy& v = *this->m_data.getLocal();
    if (v.size() <= index)
      v.resize(index + 1);
    func(v[index], rhs);
  }
};

//! Accumulator for map where accumulate does element-wise addition among
//! all entries
template<typename MapTy>
class GMapElementAccumulator: public GReducible<MapTy, ReduceMapWrap<ReduceAssignWrap<std::plus<typename MapTy::mapped_type> > > > {
  typedef ReduceAssignWrap<std::plus<typename MapTy::mapped_type> > ElementFunc;
  typedef GReducible<MapTy, ReduceMapWrap<ElementFunc> > base_type;
  typedef typename MapTy::mapped_type mapped_type;
  typedef typename MapTy::key_type key_type;

  ElementFunc func;

public:
  void update(const key_type& index, const mapped_type& rhs) {
    MapTy& v = *this->m_data.getLocal();
    func(v[index], rhs);
  }
};

//! Accumulator for T where accumulation is max
template<typename T>
class GReduceMax: public GSimpleReducible<T, gmax<T> > {
  typedef GSimpleReducible<T, gmax<T> > base_type;
public:
  GReduceMax(): base_type(gmax<T> (), std::numeric_limits<T>::min()) { }
};

//! Accumulator for T where accumulation is min
template<typename T>
class GReduceMin: public GSimpleReducible<T, gmin<T > > {
  typedef GSimpleReducible<T, gmin<T> > base_type;
public:
  GReduceMin(): base_type(gmin<T> (), std::numeric_limits<T>::max()) { }
};


//! logical AND reduction
class GReduceLogicalAND: public GSimpleReducible<bool, std::logical_and<bool> > {
  typedef GSimpleReducible<bool, std::logical_and<bool> > base_type;
public:
  GReduceLogicalAND (void): base_type (std::logical_and<bool> (), true) {}
};

class GReduceLogicalOR: public GSimpleReducible<bool, std::logical_or<bool> > {
  typedef GSimpleReducible<bool, std::logical_or<bool> > base_type;
public:
  GReduceLogicalOR (void): base_type (std::logical_or<bool> (), false) {}

};

}
#endif
