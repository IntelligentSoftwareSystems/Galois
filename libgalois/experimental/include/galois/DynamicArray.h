/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_DYNAMIC_ARRAY_H
#define GALOIS_DYNAMIC_ARRAY_H

#include <cassert>

#include <boost/noncopyable.hpp>
#include <vector>

namespace galois {

// TODO: dynamic Array using Fixed Size Allocator

template <typename T, typename A = std::allocator<T>>
class LazyDynamicArray : boost::noncopyable {

protected:
  A m_alloc;

  T* m_array;
  T* m_size;

public:
  using value_type             = T;
  using reference              = T&;
  using size_type              = size_t;
  using difference_type        = ptrdiff_t;
  using const_reference        = const value_type&;
  using pointer                = value_type*;
  using const_pointer          = const value_type*;
  using iterator               = pointer;
  using const_iterator         = const_pointer;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  explicit LazyDynamicArray(size_t length, const A& alloc = A())
      : m_alloc(alloc), m_array(nullptr), m_size(nullptr) {

    m_array = m_alloc.allocate(length);
    m_size  = m_array + length;

    assert(m_array != nullptr);
  }

  ~LazyDynamicArray() {
    m_alloc.deallocate(m_array, m_size - m_array);
    m_array = nullptr;
    m_size  = nullptr;
  }

  size_type size() const {
    assert(m_size >= m_array);
    return m_size - m_array;
  }

  template <typename... Args>
  void initialize(const size_type i, Args&&... args) {
    assert(i < size());
    m_alloc.construct(m_array + i, std::forward<Args>(args)...);
  }

  reference operator[](const size_type i) {
    assert(i < size());
    return m_array[i];
  }

  const_reference operator[](const size_type i) const {
    assert(i < size());
    return m_array[i];
  }

  // iterators:
  iterator begin() { return &m_array[0]; }
  const_iterator begin() const { return &m_array[0]; }
  iterator end() { return &m_array[size()]; }
  const_iterator end() const { return &m_array[size()]; }

  reverse_iterator rbegin() { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }
  const_reverse_iterator crbegin() const { return rbegin(); }
  const_reverse_iterator crend() const { return rend(); }
};

} // end namespace galois

#endif // GALOIS_DYNAMIC_ARRAY_H
