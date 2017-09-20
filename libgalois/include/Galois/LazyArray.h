/** Lazy Static Array -*- C++ -*-
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
 * @section Description
 * 
 * Implements something close to std::array, but which does not initialize its
 * elements.  It is the user's responsibility to make sure memory is properly
 * initialized before using.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_LAZYARRAY_H
#define GALOIS_LAZYARRAY_H

#include "Galois/LazyObject.h"

#include <iterator>
#include <stdexcept>
#include <cstddef>
#include <algorithm>
#include <utility>
#include <type_traits>

namespace galois {

/**
 * This is a container that encapsulates space for a constant size array.  The
 * initialization and destruction of items is explicitly under the control of
 * the user.
 */
template<typename _Tp, unsigned _Size>
class LazyArray {
  typedef typename std::aligned_storage<sizeof(_Tp), std::alignment_of<_Tp>::value>::type CharData;

  LazyObject<_Tp> data_[(_Size > 0 ? _Size : 1)];

  _Tp* get(size_t __n) { return &data_[__n].get(); }
  const _Tp* get(size_t __n) const { return &data_[__n].get(); }

public:
  typedef _Tp value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef pointer iterator;
  typedef const_pointer const_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  //iterators:
  iterator begin() { return iterator(get(0)); }
  const_iterator begin() const { return const_iterator(get(0)); }
  iterator end() { return iterator(get(_Size)); }
  const_iterator end() const { return const_iterator(get(_Size)); }

  reverse_iterator rbegin() { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }
  const_reverse_iterator crbegin() const { return rbegin(); }
  const_reverse_iterator crend() const { return rend(); }

  //capacity:
  size_type size() const { return _Size; }
  size_type max_size() const { return _Size; }
  bool empty() const { return _Size == 0; }

  //element access:
  reference operator[](size_type __n) { return *get(__n); }
  const_reference operator[](size_type __n) const { return *get(__n); }
  reference at(size_type __n) {
    if (__n >= _Size)
      throw std::out_of_range("lazyArray::at");
    return get(__n);
  }
  const_reference at(size_type __n) const {
    if (__n >= _Size)
      throw std::out_of_range("lazyArray::at");
    return get(__n);
  }  

  reference front() { return *get(0); }
  const_reference front() const { return *get(0); }
  reference back() { return *get(_Size > 0 ? _Size - 1 : 0); }
  const_reference back() const { return *get(_Size > 0 ? _Size - 1 : 0); }

  pointer data() { return get(0); }
  const_pointer data() const { return get(0); }

  //missing: fill swap

  template<typename... Args>
  pointer emplace(size_type __n, Args&&... args) { return new (get(__n)) _Tp(std::forward<Args>(args)...); }

  pointer construct(size_type __n, const _Tp& val) { return emplace(__n, val); }
  pointer construct(size_type __n, _Tp&& val) { return emplace(__n, std::move(val)); }

  void destroy(size_type __n) { (get(__n))->~_Tp(); }
};

}
#endif // GALOIS_LAZYARRAY_H
