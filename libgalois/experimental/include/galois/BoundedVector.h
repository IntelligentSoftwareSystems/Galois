/** Bounded Vector-*- C++ -*-
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
 * @author ahassaan@ices.utexas.edu
 */
#ifndef GALOIS_BOUNDED_VECTOR_H
#define GALOIS_BOUNDED_VECTOR_H

#include "galois/LazyArray.h"

#include <cassert>

namespace galois {

template <typename T, const size_t SZ>
class BoundedVector {

  typedef LazyArray<T, SZ> LArray;

  LArray m_array;
  size_t m_size;

  void assertValidSize () const { assert (m_size <= SZ); }

public:

  using value_type = T;
  using reference = T&;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using const_reference = const  value_type&;
  using pointer = value_type*;
  using const_pointer = const  value_type*;
  using iterator = pointer;
  using const_iterator = const_pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  BoundedVector () : m_array (), m_size (0) {}

  ~BoundedVector (void) {
    clear ();
  }

  bool empty () const { return m_size == 0; }

  bool full () const { return m_size == SZ; }

  size_type size () const { return m_size; }

  static size_type capacity () { return SZ; }

  reference operator [] (const size_type i) {
    assert (i <= size ());
    return m_array[i];
  }

  const_reference operator [] (const size_type i) const {
    assert (i <= size ());
    return m_array[i];
  }

  template <typename... Args>
  void emplace_back(Args&&... args) {
    assertValidSize ();
    assert (!full ());

    m_array.construct (m_size, std::forward<Args>(args)...);
    ++m_size;
  }

  void push_back (const_reference v) {
    assertValidSize ();
    assert (!full ());

    m_array.construct (m_size, v);
    ++m_size;
  }

  const_reference front () const { 
    return (*this)[0];
  }

  reference front () {
    return (*this)[0];
  }

  const_reference back () const {
    return (*this)[size () - 1];
  }

  reference back () {
    return (*this)[size () - 1];
  }

  void pop_back () {
    assertValidSize ();
    assert (!empty ());

    --m_size;
    assertValidSize ();
    m_array.destroy (m_size);
  }

  void clear (void) {
    for (size_t i = 0; i < size (); ++i) {
      m_array.destroy (m_size);
    }
    m_size = 0;
  }

  //iterators:
  iterator begin() { return &m_array[0]; }
  const_iterator begin() const { return &m_array[0]; }
  iterator end() { return &m_array[size ()]; }
  const_iterator end() const { return &m_array[size ()]; }

  reverse_iterator rbegin() { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }
  const_reverse_iterator crbegin() const { return rbegin(); }
  const_reverse_iterator crend() const { return rend(); }


};


template <typename T>
class DynamicBoundedVector {

  T* const m_data;
  T* const m_capacity;
  T* m_size;

public:

  using value_type = T;
  using reference = T&;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using const_reference = const  value_type&;
  using pointer = value_type*;
  using const_pointer = const  value_type*;
  using iterator = pointer;
  using const_iterator = const_pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  DynamicBoundedVector (T* beg, T* end)
    :
      m_data (beg),
      m_capacity (end),
      m_size (m_data)
  {}

  void clear (void) {
    for (T* i = m_data; i < m_size; ++i) {
      i->~T ();
    }
    m_size = m_data;
  }

  ~DynamicBoundedVector (void) {
    clear ();
    // m_data = nullptr;
    // m_capacity = nullptr;
    // m_size = nullptr;
  }

  bool empty (void) const {
    return m_size == m_data;
  }

  bool full (void) const {
    return m_size == m_capacity;
  }

  size_t size (void) const {
    return m_size - m_data;
  }

  size_t capacity (void) const {
    return m_capacity - m_data;
  }

  template <typename... Args>
  void emplace_back(Args&&... args) {
    assert (!full ());
    assert (m_size < m_capacity);

    ::new (m_size) T (std::forward<Args> (args)...);
    ++m_size;
  }

  void push_back (const T& x) {
    emplace_back (x);
  }

  void pop_back (void) {
    assert (!empty ());
    --m_size;
    m_size->~T ();
  }

  reference front (void) {
    assert (!empty ());
    return *m_data;
  }

  const_reference front (void) const {
    return const_cast<DynamicBoundedVector*> (this)->front ();
  }

  reference back (void) {
    assert (!empty ());
    return *(m_size - 1);
  }

  const_reference back (void) const {
    return const_cast<DynamicBoundedVector*> (this)->back ();
  }

  reference operator [] (const size_type i) {
    assert (i < size ());
    return *(m_data + i);
  }

  const_reference operator [] (const size_type i) const {
    return const_cast<DynamicBoundedVector&> (*this)[i];
  }

  iterator begin() { return m_data; }
  const_iterator begin() const { return m_data; }
  iterator end() { return m_size; }
  const_iterator end() const { return m_size; }

  reverse_iterator rbegin() { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }
  const_reverse_iterator crbegin() const { return rbegin(); }
  const_reverse_iterator crend() const { return rend(); }


};



} // end namespace galois

#endif // GALOIS_BOUNDED_VECTOR_H

