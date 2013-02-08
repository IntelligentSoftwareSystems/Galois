/** Large array types -*- C++ -*-
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
 * @section Description
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_LARGEARRAY_H
#define GALOIS_LARGEARRAY_H

#include "Galois/gstl.h"
#include "Galois/Runtime/mm/Mem.h"

#include <boost/utility.hpp>

namespace Galois {

/**
 * Large array of objects with proper specialization for void type. Lazy
 * template parameter indicates whether allocate() also constructs objects
 * and whether the destructor for this collection also calls destroy().
 */
template<typename T, bool isLazy>
class LargeArray: boost::noncopyable {
  T* m_data;
  size_t m_size;
public:
  typedef T value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef pointer iterator;
  typedef const_pointer const_iterator;
  const static bool has_value = true;

  LargeArray(): m_data(0), m_size(0) { }
  explicit LargeArray(size_t n): m_data(0), m_size(0) {
    allocate(n);
  }

  ~LargeArray() {
    if (isLazy)
      destroy();
    deallocate();
  }
  
  const_reference at(difference_type x) const { return m_data[x]; }
  reference at(difference_type x) { return m_data[x]; }
  const_reference operator[](size_type x) const { return m_data[x]; }
  void set(difference_type x, const_reference v) { m_data[x] = v; }
  size_type size() const { return m_size; }
  iterator begin() { return m_data; }
  const_iterator begin() const { return m_data; }
  iterator end() { return m_data + m_size; }
  const_iterator end() const { return m_data + m_size; }

  void allocate(size_type n) {
    assert(!m_data);
    m_size = n;
    m_data = reinterpret_cast<T*>(Galois::Runtime::MM::largeInterleavedAlloc(sizeof(T) * n));
    if (!isLazy)
      construct();
  }
  
  void construct() {
    for (T* ii = m_data, *ei = m_data + m_size; ii != ei; ++ii)
      new (ii) T;
  }

  void deallocate() {
    if (!m_data) return;
    Galois::Runtime::MM::largeInterleavedFree(m_data, sizeof(T) * m_size);
    m_data = 0;
    m_size = 0;
  }

  void destroy() {
    if (!m_data) return;
    uninitialized_destroy(m_data, m_data + m_size);
  }

  template<typename It>
  void copyIn(It begin, It end) {
    std::copy(begin, end, data());
  }

  // The following methods are not shared with void specialization
  reference operator[](size_type x) { return m_data[x]; }
  const_pointer data() const { return m_data; }
  pointer data() { return m_data; }
};

//! Void specialization
template<bool isLazy>
class LargeArray<void,isLazy>: boost::noncopyable {
public:
  typedef char value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef pointer iterator;
  typedef const_pointer const_iterator;
  const static bool has_value = false;

  //const_reference at(difference_type x) const { return 0; }
  //reference at(difference_type x) { return 0; }
  //const_reference operator[](size_type x) const { return 0; }
  size_type size() const { return 0; }
  iterator begin() { return 0; }
  const_iterator begin() const { return 0; }
  iterator end() { return 0; }
  const_iterator end() const { return 0; }

  void set(difference_type x, const_reference v) { }
  
  void allocate(size_type n) { }
  void construct() { }
  void deallocate() { }
  void destroy() { }

  template<typename It>
  void copyIn(It begin, It end) { }
};

}
#endif

