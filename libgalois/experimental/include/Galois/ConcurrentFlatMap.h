/** STL style map using sorted vectors -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
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
#ifndef GALOIS_CONCURRENTFLATMAP_H
#define GALOIS_CONCURRENTFLATMAP_H

#include "Galois/Bag.h"
#include "Galois/FlatMap.h"
#include "Galois/gdeque.h"

#include <boost/iterator/transform_iterator.hpp>

#include <algorithm>
#include <deque>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>
#include <deque>

namespace Galois {

namespace detail {

// TODO(ddn): super slow right now
// TODO(ddn): implement gc of log
// TODO(ddn): implement allocator
template<
  class _Key,
  class _Tp,
  class _Compare = std::less<_Key>
  >
class concurrent_flat_map_impl {
public:
  typedef _Key key_type;
  typedef _Tp mapped_type;
  typedef std::pair<_Key, _Tp> value_type;
  typedef _Compare key_compare;

protected:
  typedef concurrent_flat_map_impl base_type;

  enum class Op { INSERT, ERASE, CLEAR };

  struct Operation {
    value_type* value;
    Op type;

    Operation(Op t, value_type* v): value(v), type(t) { }
  };

  typedef value_type* value_type_pointer;
  typedef std::pair<key_type, value_type_pointer> local_map_value_type;
  typedef std::allocator<local_map_value_type> local_map_allocator_type;
  typedef Galois::flat_map<key_type, value_type_pointer, _Compare, local_map_allocator_type, Galois::gdeque<local_map_value_type>> LocalMapTy;
  //typedef Galois::flat_map<key_type, value_type_pointer, _Compare, local_map_allocator_type, std::vector<local_map_value_type>> LocalMapTy;
  typedef Galois::gdeque<Operation> LogTy;
  //typedef std::deque<Operation> LogTy;
  typedef Galois::InsertBag<value_type> ValuesTy;
  typedef Substrate::PaddedLock<true> LockTy;

  struct PerThread {
    LocalMapTy localMap;
    size_t lastVersion;
    PerThread(): lastVersion(0) { }
    PerThread(const key_compare& c, const local_map_allocator_type& a): localMap(c, a), lastVersion(0) { }
  };

  template<typename IteratorTy, typename ResultTy>
  struct Dereference: std::unary_function<typename std::iterator_traits<IteratorTy>::reference, ResultTy> {
    ResultTy operator()(typename std::iterator_traits<IteratorTy>::reference x) const {
      return *x.second;
    }
  };

  key_compare comp;
  Substrate::PerThreadStorage<PerThread> pts;
  ValuesTy values;
  LockTy logLock;
  LogTy log;
  std::atomic<size_t> logVersion;

  void move_assign(concurrent_flat_map_impl& lhs, concurrent_flat_map_impl&& rhs) {
    std::swap(lhs.comp, rhs.comp);
    std::swap(lhs.pts, rhs.pts);
    std::swap(lhs.values, rhs.values);
    std::swap(lhs.log, rhs.log);
    lhs.logVersion = rhs.logVersion.load();
  }

  void applyOperation(PerThread& pt, Operation& op) {
    switch (op.type) {
      case Op::INSERT: 
        pt.localMap.emplace(std::piecewise_construct, std::forward_as_tuple(op.value->first), std::forward_as_tuple(op.value)); 
        break;
      case Op::ERASE:
        pt.localMap.erase(op.value->first);
        break;
      case Op::CLEAR:
        pt.localMap.clear();
        break;
      default: abort();
    };
  }

  int slowErase(PerThread& pt, const key_type& k) {
    std::unique_lock<LockTy> ll(logLock, std::defer_lock);
    do {
      fastSync(pt);
    } while (!ll.try_lock());
    fastSync(pt); // fully synchronized since we hold the lock
    auto ii = pt.localMap.find(k);
    if (ii != pt.localMap.end()) {
      pt.localMap.erase(ii);
      log.emplace_back(Op::ERASE, ii->second);
      assert(logVersion == pt.lastVersion);
      logVersion += 1;
      pt.lastVersion = ++logVersion;
      return 1;
    }
    return 0;
  }

  template<typename... Args>
  typename LocalMapTy::iterator slowInsert(PerThread& pt, bool& inserted, Args&&... args) {
    std::unique_lock<LockTy> ll(logLock, std::defer_lock);
    do {
      fastSync(pt);
    } while (!ll.try_lock());
    fastSync(pt); // fully synchronized since we hold the lock
    value_type* v = &values.emplace(std::forward<Args>(args)...);
    auto p = pt.localMap.emplace(std::piecewise_construct, std::forward_as_tuple(v->first), std::forward_as_tuple(v));
    inserted = p.second;
    if (inserted) {
      log.emplace_back(Op::INSERT, v);
      assert(logVersion == pt.lastVersion);
      pt.lastVersion = ++logVersion;
    } else {
      values.pop();
    }

    return p.first;
  }

  //! Quick and dirty synchronization with log; returns true if something
  //! changed
  bool fastSync(PerThread& pt) { 
    size_t ll = logVersion.load();
    if (pt.lastVersion == ll)
      return false;
    auto ii = log.begin();
    std::advance(ii, pt.lastVersion);
    for (; pt.lastVersion < ll; ++pt.lastVersion, ++ii) {
      assert(ii != log.end());
      applyOperation(pt, *ii);
    }
    return true;
  }

  bool fastSync(PerThread& pt) const { return false; }

  void slowSync() { 
    std::unique_lock<LockTy> ll(logLock, std::defer_lock);
    PerThread& pt = *pts.getLocal();
    do {
      fastSync(pt);
    } while (!ll.try_lock());
    fastSync(pt); // fully synchronized since we hold the lock
  }

  void slowSync() const { }

  bool key_eq(const key_type& k1, const key_type& k2) const {
    return !key_comp()(k1, k2) && !key_comp()(k2, k1);
  }

public:
  typedef typename ValuesTy::pointer pointer;
  typedef typename ValuesTy::const_pointer const_pointer;
  typedef typename ValuesTy::reference reference;
  typedef typename ValuesTy::const_reference const_reference;
  typedef boost::transform_iterator<Dereference<typename LocalMapTy::iterator, reference>, typename LocalMapTy::iterator> iterator;
  typedef boost::transform_iterator<Dereference<typename LocalMapTy::const_iterator, const_reference>, typename LocalMapTy::const_iterator> const_iterator;
  typedef typename LocalMapTy::size_type size_type;
  typedef typename LocalMapTy::difference_type difference_type;
  typedef boost::transform_iterator<Dereference<typename LocalMapTy::reverse_iterator, reference>, typename LocalMapTy::reverse_iterator> reverse_iterator;
  typedef boost::transform_iterator<Dereference<typename LocalMapTy::const_reverse_iterator, const_reference>, typename LocalMapTy::const_reverse_iterator> const_reverse_iterator;

  iterator begin() noexcept { slowSync(); return iterator { pts.getLocal()->localMap.begin() }; }
  const_iterator begin() const noexcept { slowSync(); return const_iterator { pts.getLocal()->localMap.begin() }; }
  iterator end() noexcept { slowSync(); return iterator { pts.getLocal()->localMap.end() }; }
  const_iterator end() const noexcept { slowSync(); return const_iterator { pts.getLocal()->localMap.end() }; }
  reverse_iterator rbegin() noexcept { slowSync(); return reverse_iterator { pts.getLocal()->localMap.rbegin() }; }
  const_reverse_iterator rbegin() const noexcept { slowSync(); return const_reverse_iterator { pts.getLocal()->localMap.rbegin() }; }
  reverse_iterator rend() noexcept { slowSync(); return reverse_iterator { pts.getLocal()->localMap.rend() }; }
  const_reverse_iterator  rend() const noexcept { slowSync(); return const_reverse_iterator { pts.getLocal()->localMap.rend() }; }
  const_iterator cbegin() const noexcept { slowSync(); return const_iterator { pts.getLocal()->localMap.begin() }; }
  const_iterator cend() const noexcept { slowSync(); return const_iterator { pts.getLocal()->localMap.end() }; }
  const_reverse_iterator crbegin() const noexcept { slowSync(); return const_reverse_iterator { pts.getLocal()->localMap.rbegin() }; }
  const_reverse_iterator crend() const noexcept { slowSync(); return const_reverse_iterator { pts.getLocal()->localMap.rend() }; }

  concurrent_flat_map_impl(const key_compare& c): comp(c), pts(c, local_map_allocator_type()), logVersion(0) { }

  concurrent_flat_map_impl(concurrent_flat_map_impl&& o) {
    move_assign(*this, std::move(o));
  }

  concurrent_flat_map_impl& operator=(concurrent_flat_map_impl&& o) {
    move_assign(*this, std::move(o));
    return *this;
  }

  bool empty() const noexcept { slowSync(); return pts.getLocal()->localMap.empty(); }
  size_type size() const noexcept { slowSync(); return pts.getLocal()->localMap.size(); }
  size_type max_size() const noexcept { return pts.getLocal()->localMap.max_size(); }

  template<typename KeyTy>
  mapped_type& operator[](KeyTy&& k) {
    PerThread& pt = *pts.getLocal();
    auto ii = pt.localMap.lower_bound(k);
    for (int i = 0; i < 2; ++i, ii = pt.localMap.lower_bound(k)) {
      if (ii != pt.localMap.end() && key_eq(k, ii->first))
        return ii->second->second;
      else if (i == 0 && !fastSync(pt))
        ;
      else
        break;
    }
    bool inserted;
    ii = slowInsert(pt, inserted,
        std::piecewise_construct,
        std::forward_as_tuple(std::forward<KeyTy>(k)),
        std::tuple<>());
    return ii->second->second;
  }

  mapped_type& at(const key_type& k) {
    PerThread& pt = *pts.getLocal();
    auto ii = pt.localMap.lower_bound(k);
    for (int i = 0; i < 2; ++i, ii = pt.localMap.lower_bound(k)) {
      if (ii != pt.localMap.end() && key_eq(k, ii->first))
        return ii->second->second;
      else if (i == 0 && !fastSync(pt))
          ;
      else
        break;
    }
    throw std::out_of_range("concurrent_flat_map::at");
  }

  const mapped_type& at(const key_type& k) const {
    PerThread& pt = *pts.getLocal();
    auto ii = pt.localMap.lower_bound(k);
    for (int i = 0; i < 2; ++i, ii = pt.localMap.lower_bound(k)) {
      if (ii != pt.localMap.end() && key_eq(k, ii->first))
        return ii->second->second;
      else if (i == 0 && !fastSync(pt))
          ;
      else
        break;
    }
    throw std::out_of_range("concurrent_flat_map::at");
  }

  template<typename PairTy>
  std::pair<iterator, bool> insert(PairTy&& x) {
    PerThread& pt = *pts.getLocal();
    auto ii = pt.localMap.lower_bound(x.first);
    for (int i = 0; i < 2; ++i, ii = pt.localMap.lower_bound(x.first)) {
      if (ii != pt.localMap.end() && key_eq(x.first, ii->first))
        return std::make_pair(iterator { ii }, false);
      else if (i == 0 && !fastSync(pt))
        ;
      else
        break;
    }
    bool inserted;
    ii = slowInsert(pt, inserted, std::forward<PairTy>(x));
    return std::make_pair(iterator { ii }, inserted); 
  }

  template<typename InputIteratorTy>
  void insert(InputIteratorTy first, InputIteratorTy last) {
    PerThread& pt = *pts.getLocal();
    bool inserted;
    for (; first != last; ++first) {
      slowInsert(pt, inserted, *first);
    }
  }

  size_type erase(const key_type& k) {
    PerThread& pt = *pts.getLocal();
    auto ii = pt.localMap.lower_bound(k);
    for (int i = 0; i < 2; ++i, ii = pt.localMap.lower_bound(k)) {
      // TODO(ddn): Could defer syncing
      if (ii != pt.localMap.end() && key_eq(k, ii->first))
        return slowErase(pt, k);
      else if (i == 0 && !fastSync(pt))
        ;
      else
        break;
    }
    return 0;
  }

  iterator find(const key_type& k) {
    PerThread& pt = *pts.getLocal();
    auto ii = pt.localMap.lower_bound(k);
    for (int i = 0; i < 2; ++i, ii = pt.localMap.lower_bound(k)) {
      if (ii != pt.localMap.end() && key_eq(k, ii->first))
        return iterator { ii };
      else if (i == 0 && !fastSync(pt))
        ;
      else
        break;
    }
    return iterator { pt.localMap.end() };
  }

  const_iterator find(const key_type& k) const {
    PerThread& pt = *pts.getLocal();
    auto ii = pt.localMap.lower_bound(k);
    for (int i = 0; i < 2; ++i, ii = pt.localMap.lower_bound(k)) {
      if (ii != pt.localMap.end() && key_eq(k.first, ii->first))
        return const_iterator { ii };
      else if (i == 0 && !fastSync(pt))
        ;
      else
        break;
    }
    return const_iterator { pt.localMap.end() };
  }

  void clear() {
    std::unique_lock<LockTy> ll(logLock);
    PerThread& pt = *pts.getLocal();
    pt.localMap.clear();
    log.emplace_back(Op::CLEAR, nullptr);
    pt.lastVersion = ++logVersion;
  }

  iterator lower_bound(const key_type& k) {
    slowSync();
    return iterator { pts.getLocal()->localMap->lower_bound(k) };
  }

  const_iterator lower_bound(const key_type& k) const {
    slowSync();
    return const_iterator { pts.getLocal()->localMap->lower_bound(k) };
  }

  iterator upper_bound(const key_type& k) {
    slowSync();
    return iterator { pts.getLocal()->localMap->upper_bound(k) };
  }

  const_iterator upper_bound(const key_type& k) const {
    slowSync();
    return const_iterator { pts.getLocal()->localMap->upper_bound(k) };
  }

  key_compare key_comp() const { return comp; }
};


}

/**
 * Concurrent version of {@link flat_map} using operation logs to reduce
 * synchronization. Designed for read-heavy workloads.
 *
 * Operations on individual entries like {@link at()} or {@link erase()} are
 * not linearizable with respect to other operations on individual entries.
 * Insertions are however linearizable with respect to other insertions.
 *
 * Operations on the entire map like {@link size()} or {@link begin()} are
 * linearizable with respect to other operations on the entire map.
 */
template<
  class _Key,
  class _Tp,
  class _Compare = std::less<_Key>,
  bool _HasCopy = std::is_copy_constructible<_Tp>::value
  >
struct concurrent_flat_map: public detail::concurrent_flat_map_impl<_Key, _Tp, _Compare> {
  typedef typename concurrent_flat_map::base_type base;

  concurrent_flat_map(const _Compare& c = _Compare()): base(c) { }

  concurrent_flat_map(const concurrent_flat_map& o): base(o.key_comp()) {
    this->insert(o.begin(), o.end());
  }

  concurrent_flat_map& operator=(const concurrent_flat_map& o) {
    if (this != &o) {
      concurrent_flat_map tmp(o);
      this->move_assign(*this, std::move(tmp));
    }
    return *this;
  }

  template<typename InputIteratorTy>
  concurrent_flat_map(InputIteratorTy first, InputIteratorTy last, const _Compare& c = _Compare()): base(c)
  {
    this->insert(first, last);
  }
};

template<
  class _Key,
  class _Tp,
  class _Compare
  >
struct concurrent_flat_map<_Key, _Tp, _Compare, false>: public detail::concurrent_flat_map_impl<_Key, _Tp, _Compare> {
  typedef typename concurrent_flat_map::base_type base;

  concurrent_flat_map(const _Compare& c = _Compare()): base(c) { }

  template<typename InputIteratorTy>
  concurrent_flat_map(InputIteratorTy first, InputIteratorTy last, const _Compare& c = _Compare()): base(c)
  {
    this->insert(first, last);
  }
};

}

#endif
