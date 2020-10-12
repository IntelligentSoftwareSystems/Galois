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

#ifndef GALOIS_LARGEARRAY_H
#define GALOIS_LARGEARRAY_H

#include <iostream>
#include <utility>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>

#include "galois/config.h"
#include "galois/Galois.h"
#include "galois/gIO.h"
#include "galois/ParallelSTL.h"
#include "galois/runtime/Mem.h"
#include "galois/substrate/NumaMem.h"

namespace galois {

namespace runtime {
extern unsigned activeThreads;
} // end namespace runtime

/**
 * Large array of objects with proper specialization for void type and
 * supporting various allocation and construction policies.
 *
 * @tparam T value type of container
 */
template <typename T>
class LargeArray {
  substrate::LAptr m_realdata;
  T* m_data;
  size_t m_size;

public:
  typedef T raw_value_type;
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

  // Extra indirection to support incomplete T's
  struct size_of {
    const static size_t value = sizeof(T);
  };

protected:
  enum AllocType { Blocked, Local, Interleaved, Floating };
  void allocate(size_type n, AllocType t) {
    assert(!m_data);
    m_size = n;
    switch (t) {
    case Blocked:
      m_realdata =
          substrate::largeMallocBlocked(n * sizeof(T), runtime::activeThreads);
      break;
    case Interleaved:
      m_realdata = substrate::largeMallocInterleaved(n * sizeof(T),
                                                     runtime::activeThreads);
      break;
    case Local:
      m_realdata = substrate::largeMallocLocal(n * sizeof(T));
      break;
    case Floating:
      m_realdata = substrate::largeMallocFloating(n * sizeof(T));
      break;
    };
    m_data = reinterpret_cast<T*>(m_realdata.get());
  }

private:
  /*
   * To support boost serialization
   */
  friend class boost::serialization::access;
  template <typename Archive>
  void save(Archive& ar, const unsigned int) const {

    // TODO DON'T USE CERR
    // std::cerr << "save m_size : " << m_size << " Threads : " <<
    // runtime::activeThreads << "\n";
    ar << m_size;
    // for(size_t i = 0; i < m_size; ++i){
    // ar << m_data[i];
    //}
    ar << boost::serialization::make_binary_object(m_data, m_size * sizeof(T));
    /*
     * Cas use make_array too as shown below
     * IMPORTANT: Use make_array as temp fix for benchmarks using non-trivial
     * structures in nodeData (Eg. SGD) This also requires changes in
     * libgalois/include/galois/graphs/Details.h (specified in the file).
     */
    // ar << boost::serialization::make_array<T>(m_data, m_size);
  }
  template <typename Archive>
  void load(Archive& ar, const unsigned int) {
    ar >> m_size;

    // TODO DON'T USE CERR
    // std::cerr << "load m_size : " << m_size << " Threads : " <<
    // runtime::activeThreads << "\n";

    // TODO: For now, always use allocateInterleaved
    // Allocates and sets m_data pointer
    if (!m_data)
      allocateInterleaved(m_size);

    // for(size_t i = 0; i < m_size; ++i){
    // ar >> m_data[i];
    //}
    ar >> boost::serialization::make_binary_object(m_data, m_size * sizeof(T));
    /*
     * Cas use make_array too as shown below
     * IMPORTANT: Use make_array as temp fix for SGD
     *            This also requires changes in
     * libgalois/include/galois/graphs/Details.h (specified in the file).
     */
    // ar >> boost::serialization::make_array<T>(m_data, m_size);
  }
  // The macro BOOST_SERIALIZATION_SPLIT_MEMBER() generates code which invokes
  // the save or load depending on whether the archive is used for saving or
  // loading
  BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
  /**
   * Wraps existing buffer in LargeArray interface.
   */
  LargeArray(void* d, size_t s) : m_data(reinterpret_cast<T*>(d)), m_size(s) {}

  LargeArray() : m_data(0), m_size(0) {}

  LargeArray(LargeArray&& o) : m_data(0), m_size(0) {
    std::swap(this->m_realdata, o.m_realdata);
    std::swap(this->m_data, o.m_data);
    std::swap(this->m_size, o.m_size);
  }

  LargeArray& operator=(LargeArray&& o) {
    std::swap(this->m_realdata, o.m_realdata);
    std::swap(this->m_data, o.m_data);
    std::swap(this->m_size, o.m_size);
    return *this;
  }

  LargeArray(const LargeArray&) = delete;
  LargeArray& operator=(const LargeArray&) = delete;

  ~LargeArray() {
    destroy();
    deallocate();
  }

  friend void swap(LargeArray& lhs, LargeArray& rhs) {
    std::swap(lhs.m_realdata, rhs.m_realdata);
    std::swap(lhs.m_data, rhs.m_data);
    std::swap(lhs.m_size, rhs.m_size);
  }

  const_reference at(difference_type x) const { return m_data[x]; }
  reference at(difference_type x) { return m_data[x]; }
  const_reference operator[](size_type x) const { return m_data[x]; }
  reference operator[](size_type x) { return m_data[x]; }
  void set(difference_type x, const_reference v) { m_data[x] = v; }
  size_type size() const { return m_size; }
  iterator begin() { return m_data; }
  const_iterator begin() const { return m_data; }
  iterator end() { return m_data + m_size; }
  const_iterator end() const { return m_data + m_size; }

  //! [allocatefunctions]
  //! Allocates interleaved across NUMA (memory) nodes.
  void allocateInterleaved(size_type n) { allocate(n, Interleaved); }

  /**
   * Allocates using blocked memory policy
   *
   * @param  n         number of elements to allocate
   */
  void allocateBlocked(size_type n) { allocate(n, Blocked); }

  /**
   * Allocates using Thread Local memory policy
   *
   * @param  n         number of elements to allocate
   */
  void allocateLocal(size_type n) { allocate(n, Local); }

  /**
   * Allocates using no memory policy (no pre alloc)
   *
   * @param  n         number of elements to allocate
   */
  void allocateFloating(size_type n) { allocate(n, Floating); }

  /**
   * Allocate memory to threads based on a provided array specifying which
   * threads receive which elements of data.
   *
   * @tparam RangeArrayTy The type of the threadRanges array; should either
   * be uint32_t* or uint64_t*
   * @param numberOfElements Number of elements to allocate space for
   * @param threadRanges An array specifying how elements should be split
   * among threads
   */
  template <typename RangeArrayTy>
  void allocateSpecified(size_type numberOfElements,
                         RangeArrayTy& threadRanges) {
    assert(!m_data);

    m_realdata = substrate::largeMallocSpecified(numberOfElements * sizeof(T),
                                                 runtime::activeThreads,
                                                 threadRanges, sizeof(T));

    m_size = numberOfElements;
    m_data = reinterpret_cast<T*>(m_realdata.get());
  }
  //! [allocatefunctions]

  template <typename... Args>
  void construct(Args&&... args) {
    for (T *ii = m_data, *ei = m_data + m_size; ii != ei; ++ii)
      new (ii) T(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void constructAt(size_type n, Args&&... args) {
    new (&m_data[n]) T(std::forward<Args>(args)...);
  }

  //! Allocate and construct
  template <typename... Args>
  void create(size_type n, Args&&... args) {
    allocateInterleaved(n);
    construct(std::forward<Args>(args)...);
  }

  void deallocate() {
    m_realdata.reset();
    m_data = 0;
    m_size = 0;
  }

  void destroy() {
    if (!m_data)
      return;
    galois::ParallelSTL::destroy(m_data, m_data + m_size);
  }

  template <typename U = T>
  std::enable_if_t<!std::is_scalar<U>::value> destroyAt(size_type n) {
    (&m_data[n])->~T();
  }

  template <typename U = T>
  std::enable_if_t<std::is_scalar<U>::value> destroyAt(size_type) {}

  // The following methods are not shared with void specialization
  const_pointer data() const { return m_data; }
  pointer data() { return m_data; }
};

//! Void specialization
template <>
class LargeArray<void> {

private:
  /*
   * To support boost serialization
   * Can use single function serialize instead of save and load, since both save
   * and load have identical code.
   */
  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive&, const unsigned int) const {}

public:
  LargeArray(void*, size_t) {}
  LargeArray()                  = default;
  LargeArray(const LargeArray&) = delete;
  LargeArray& operator=(const LargeArray&) = delete;

  friend void swap(LargeArray&, LargeArray&) {}

  typedef void raw_value_type;
  typedef void* value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef value_type reference;
  typedef value_type const_reference;
  typedef value_type* pointer;
  typedef value_type* const_pointer;
  typedef pointer iterator;
  typedef const_pointer const_iterator;
  const static bool has_value = false;
  struct size_of {
    const static size_t value = 0;
  };

  const_reference at(difference_type) const { return 0; }
  reference at(difference_type) { return 0; }
  const_reference operator[](size_type) const { return 0; }
  template <typename AnyTy>
  void set(difference_type, AnyTy) {}
  size_type size() const { return 0; }
  iterator begin() { return 0; }
  const_iterator begin() const { return 0; }
  iterator end() { return 0; }
  const_iterator end() const { return 0; }

  void allocateInterleaved(size_type) {}
  void allocateBlocked(size_type) {}
  void allocateLocal(size_type, bool = true) {}
  void allocateFloating(size_type) {}
  template <typename RangeArrayTy>
  void allocateSpecified(size_type, RangeArrayTy) {}

  template <typename... Args>
  void construct(Args&&...) {}
  template <typename... Args>
  void constructAt(size_type, Args&&...) {}
  template <typename... Args>
  void create(size_type, Args&&...) {}

  void deallocate() {}
  void destroy() {}
  void destroyAt(size_type) {}

  const_pointer data() const { return 0; }
  pointer data() { return 0; }
};

} // namespace galois
#endif
