/** Bounded Vector-*- C++ -*-
 * @file
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
 *
 * @author ahassaan@ices.utexas.edu
 */
#ifndef GALOIS_BOUNDED_VECTOR_H
#define GALOIS_BOUNDED_VECTOR_H

#include "Galois/LazyArray.h"

namespace Galois {

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

} // end namespace Galois

#endif // GALOIS_BOUNDED_VECTOR_H

