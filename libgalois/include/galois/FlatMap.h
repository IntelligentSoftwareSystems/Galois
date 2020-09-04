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

#ifndef GALOIS_FLATMAP_H
#define GALOIS_FLATMAP_H

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "galois/config.h"

namespace galois {

//! Simple map data structure, based off a single array.
template <class _Key, class _Tp, class _Compare = std::less<_Key>,
          class _Alloc = std::allocator<std::pair<_Key, _Tp>>,
          class _Store = std::vector<std::pair<_Key, _Tp>, _Alloc>>
class flat_map {
public:
  typedef _Key key_type;
  typedef _Tp mapped_type;
  typedef std::pair<_Key, _Tp> value_type;
  typedef _Compare key_compare;
  typedef _Alloc allocator_type;

  class value_compare {
    friend class flat_map<_Key, _Tp, _Compare, _Alloc, _Store>;

  protected:
    _Compare comp;

    value_compare(_Compare __c) : comp(__c) {}

  public:
    bool operator()(const value_type& __x, const value_type& __y) const {
      return comp(__x.first, __y.first);
    }
  };

private:
  /// This turns...
  typedef typename _Alloc::template rebind<value_type>::other _Pair_alloc_type;

  typedef _Store _VectTy;
  _VectTy _data;
  _Compare _comp;

  class value_key_compare {
    friend class flat_map<_Key, _Tp, _Compare, _Alloc, _Store>;

  protected:
    _Compare comp;

    value_key_compare(_Compare __c) : comp(__c) {}

  public:
    bool operator()(const value_type& __x, const key_type& __y) const {
      return comp(__x.first, __y);
    }
  };

  value_key_compare value_key_comp() const {
    return value_key_compare(key_comp());
  }

  bool key_eq(const key_type& k1, const key_type& k2) const {
    return !key_comp()(k1, k2) && !key_comp()(k2, k1);
  }

  void resort() { std::sort(_data.begin(), _data.end(), value_comp()); }

public:
  typedef typename _Pair_alloc_type::pointer pointer;
  typedef typename _Pair_alloc_type::const_pointer const_pointer;
  typedef typename _Pair_alloc_type::reference reference;
  typedef typename _Pair_alloc_type::const_reference const_reference;
  typedef typename _VectTy::iterator iterator;
  typedef typename _VectTy::const_iterator const_iterator;
  typedef typename _VectTy::size_type size_type;
  typedef typename _VectTy::difference_type difference_type;
  typedef typename _VectTy::reverse_iterator reverse_iterator;
  typedef typename _VectTy::const_reverse_iterator const_reverse_iterator;

  flat_map() : _data(), _comp() {}

  explicit flat_map(const _Compare& __comp,
                    const allocator_type& = allocator_type())
      // XXX :_data(_Pair_alloc_type(__a)), _comp(__comp) {}
      : _data(), _comp(__comp) {}

  flat_map(const flat_map& __x) : _data(__x._data), _comp(__x._comp) {}

  flat_map(flat_map&& __x)
      /* noexcept(std::is_nothrow_copy_constructible<_Compare>::value) */
      : _data(std::move(__x._data)), _comp(std::move(__x._comp)) {}

  /*
  flat_map(std::initializer_list<value_type> __l,
       const _Compare& __comp = _Compare(),
       const allocator_type& __a = allocator_type())
    : _data(__l, _Pair_alloc_type(__a)), _comp(__comp) { resort(); }
   */

  template <typename _InputIterator>
  flat_map(_InputIterator __first, _InputIterator __last)
      : _data(__first, __last), _comp() {
    resort();
  }

  template <typename _InputIterator>
  flat_map(_InputIterator __first, _InputIterator __last, const _Compare&,
           const allocator_type& __a = allocator_type())
      : _data(__first, __last, _Pair_alloc_type(__a)) {
    resort();
  }

  flat_map& operator=(const flat_map& __x) {
    _data = __x._data;
    _comp = __x._comp;
    return *this;
  }

  flat_map& operator=(flat_map&& __x) {
    clear();
    swap(__x);
    return *this;
  }

  /*
  flat_map& operator=(std::initializer_list<value_type> __l) {
    clear();
    insert(__l.begin(), __l.end());
    return *this;
  }
   */

  allocator_type get_allocator() const /* noexcept */ {
    return allocator_type(_data.get_allocator());
  }

  // iterators

  iterator begin() /* noexcept */ { return _data.begin(); }
  const_iterator begin() const /* noexcept */ { return _data.begin(); }
  iterator end() /* noexcept */ { return _data.end(); }
  const_iterator end() const /* noexcept */ { return _data.end(); }
  reverse_iterator rbegin() /* noexcept */ { return _data.rbegin(); }
  const_reverse_iterator rbegin() const /* noexcept */ {
    return _data.rbegin();
  }
  reverse_iterator rend() /* noexcept */ { return _data.rend(); }
  const_reverse_iterator rend() const /* noexcept */ { return _data.rend(); }
  const_iterator cbegin() const /* noexcept */ { return _data.begin(); }
  const_iterator cend() const /* noexcept */ { return _data.end(); }
  const_reverse_iterator crbegin() const /* noexcept */ {
    return _data.rbegin();
  }
  const_reverse_iterator crend() const /* noexcept */ { return _data.rend(); }

  bool empty() const /* noexcept */ { return _data.empty(); }
  size_type size() const /* noexcept */ { return _data.size(); }
  size_type max_size() const /* noexcept */ { return _data.max_size(); }

  template <typename... Args>
  std::pair<iterator, bool> emplace(Args&&... args) {
    // assert(std::adjacent_find(_data.begin(), _data.end(), [&](const
    // value_type& a, const value_type& b) {
    //    return key_comp()(b.first, a.first);
    //}) == _data.end());
    _data.emplace_back(std::forward<Args>(args)...);
    value_type& v = _data.back();
    auto ee       = _data.end();
    --ee;
    auto __i = std::lower_bound(_data.begin(), ee, v.first, value_key_comp());
    // key < __i->first
    bool retval = __i == ee || key_comp()(v.first, (*__i).first);
    if (retval) {
      if (__i != ee) {
        value_type tmp = std::move(v);
        __i            = _data.emplace(__i, std::move(tmp));
        _data.pop_back();
      }
    } else {
      // key == __i->first
      _data.pop_back();
    }
    return std::make_pair(__i, retval);
  }

  mapped_type& operator[](const key_type& __k) {
    iterator __i = lower_bound(__k);
    // __i->first is greater than or equivalent to __k.
    if (__i == end() || key_comp()(__k, (*__i).first))
      __i = _data.emplace(__i, std::piecewise_construct,
                          std::forward_as_tuple(__k), std::tuple<>());
    return (*__i).second;
  }

  mapped_type& operator[](key_type&& __k) {
    iterator __i = lower_bound(__k);
    // __i->first is greater than or equivalent to __k.
    if (__i == end() || key_comp()(__k, (*__i).first))
      __i =
          _data.emplace(__i, std::piecewise_construct,
                        std::forward_as_tuple(std::move(__k)), std::tuple<>());
    return (*__i).second;
  }

  mapped_type& at(const key_type& __k) {
    iterator __i = lower_bound(__k);
    if (__i == end() || key_comp()(__k, (*__i).first))
      throw std::out_of_range("flat_map::at");
    return (*__i).second;
  }

  const mapped_type& at(const key_type& __k) const {
    const_iterator __i = lower_bound(__k);
    if (__i == end() || key_comp()(__k, (*__i).first))
      throw std::out_of_range("flat_map::at");
    return (*__i).second;
  }

  template <typename PairTy,
            typename = typename std::enable_if<
                std::is_constructible<value_type, PairTy&&>::value>::type>
  std::pair<iterator, bool> insert(PairTy&& __x) {
    return emplace(std::forward<PairTy>(__x));
  }

  /*
  void insert(std::initializer_list<value_type> __list) {
    insert(__list.begin(), __list.end());
  }
   */

  template <typename _InputIterator>
  void insert(_InputIterator __first, _InputIterator __last) {
    while (__first != __last)
      insert(*__first++);
  }

  iterator erase(const_iterator __position) { return _data.erase(__position); }
  iterator erase(iterator __position) { return _data.erase(__position); }

  size_type erase(const key_type& __x) {
    auto i = find(__x);
    if (i != end()) {
      _data.erase(i);
      return 1;
    }
    return 0;
  }

  iterator erase(const_iterator __first, const_iterator __last) {
    return _data.erase(__first, __last);
  }

  void swap(flat_map& __x) {
    _data.swap(__x._data);
    std::swap(_comp, __x._comp);
  }

  void clear() /* noexcept */ { _data.clear(); }

  key_compare key_comp() const { return _comp; }
  value_compare value_comp() const { return value_compare(key_comp()); }

  iterator find(const key_type& __x) {
    auto i = lower_bound(__x);
    if (i != end() && key_eq(i->first, __x))
      return i;
    return end();
  }

  const_iterator find(const key_type& __x) const {
    auto i = lower_bound(__x);
    if (i != end() && key_eq(i->first, __x))
      return i;
    return end();
  }

  size_type count(const key_type& __x) const {
    return find(__x) == end() ? 0 : 1;
  }

  iterator lower_bound(const key_type& __x) {
    return std::lower_bound(_data.begin(), _data.end(), __x, value_key_comp());
  }
  const_iterator lower_bound(const key_type& __x) const {
    return std::lower_bound(_data.begin(), _data.end(), __x, value_key_comp());
  }

  iterator upper_bound(const key_type& __x) {
    return std::upper_bound(_data.begin(), _data.end(), __x, value_key_comp());
  }
  const_iterator upper_bound(const key_type& __x) const {
    return std::upper_bound(_data.begin(), _data.end(), __x, value_key_comp());
  }

  std::pair<iterator, iterator> equal_range(const key_type& __x) {
    return std::make_pair(lower_bound(__x), upper_bound(__x));
  }

  std::pair<const_iterator, const_iterator>
  equal_range(const key_type& __x) const {
    return std::make_pair(lower_bound(__x), upper_bound(__x));
  }
};

template <typename _Key, typename _Tp, typename _Compare, typename _Alloc>
inline bool operator==(const flat_map<_Key, _Tp, _Compare, _Alloc>& __x,
                       const flat_map<_Key, _Tp, _Compare, _Alloc>& __y) {
  return __x._data == __y._data;
}

template <typename _Key, typename _Tp, typename _Compare, typename _Alloc>
inline bool operator<(const flat_map<_Key, _Tp, _Compare, _Alloc>& __x,
                      const flat_map<_Key, _Tp, _Compare, _Alloc>& __y) {
  return __x._data < __y._data;
}

/// Based on operator==
template <typename _Key, typename _Tp, typename _Compare, typename _Alloc>
inline bool operator!=(const flat_map<_Key, _Tp, _Compare, _Alloc>& __x,
                       const flat_map<_Key, _Tp, _Compare, _Alloc>& __y) {
  return !(__x == __y);
}

/// Based on operator<
template <typename _Key, typename _Tp, typename _Compare, typename _Alloc>
inline bool operator>(const flat_map<_Key, _Tp, _Compare, _Alloc>& __x,
                      const flat_map<_Key, _Tp, _Compare, _Alloc>& __y) {
  return __y < __x;
}

/// Based on operator<
template <typename _Key, typename _Tp, typename _Compare, typename _Alloc>
inline bool operator<=(const flat_map<_Key, _Tp, _Compare, _Alloc>& __x,
                       const flat_map<_Key, _Tp, _Compare, _Alloc>& __y) {
  return !(__y < __x);
}

/// Based on operator<
template <typename _Key, typename _Tp, typename _Compare, typename _Alloc>
inline bool operator>=(const flat_map<_Key, _Tp, _Compare, _Alloc>& __x,
                       const flat_map<_Key, _Tp, _Compare, _Alloc>& __y) {
  return !(__x < __y);
}

} // namespace galois

namespace std {

/// See galois::flat_map::swap().
template <typename _Key, typename _Tp, typename _Compare, typename _Alloc>
inline void swap(galois::flat_map<_Key, _Tp, _Compare, _Alloc>& __x,
                 galois::flat_map<_Key, _Tp, _Compare, _Alloc>& __y) {
  __x.swap(__y);
}

} // namespace std

#endif
