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

#ifndef GALOIS_PODRESIZEABLEARRAY_H
#define GALOIS_PODRESIZEABLEARRAY_H

#include <iterator>
#include <stdexcept>
#include <cstddef>
#include <algorithm>
#include <utility>
#include <type_traits>

#include "galois/config.h"

namespace galois {

/**
 * This is a container that encapsulates a resizeable array 
 * of plain-old-datatype (POD) elements.
 * There is no initialization or destruction of elements.
 */
template <typename _Tp>
class PODResizeableArray {
  _Tp* data_;
  size_t capacity_;
  size_t size_;

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

  PODResizeableArray() : data_(NULL), capacity_(0), size_(0) {}

  template <class InputIterator>
  PODResizeableArray(InputIterator first, InputIterator last) 
    : data_(NULL), capacity_(0), size_(0) 
  {
    size_t to_add = last - first;
    resize(to_add);
    std::copy_n(first, to_add, begin());
  }

  PODResizeableArray(size_t n) 
    : data_(NULL), capacity_(0), size_(0) 
  { 
    resize(n); 
  }

  //! disabled (shallow) copy constructor
  PODResizeableArray(const PODResizeableArray&) = delete;

  //! move constructor
  PODResizeableArray(PODResizeableArray&& v)
    : data_(v.data_), capacity_(v.capacity_), size_(v.size_) 
  {
    v.data_ = NULL;
    v.capacity_ = 0;
    v.size_ = 0;
  }

  //! disabled (shallow) copy assignment operator
  PODResizeableArray& operator=(const PODResizeableArray&) = delete;

  //! move assignment operator
  PODResizeableArray& operator=(PODResizeableArray&& v) {
    if (data_ != NULL) free(data_);
    data_ = v.data_;
    capacity_ = v.capacity_;
    size_ = v.size_;
    v.data_ = NULL;
    v.capacity_ = 0;
    v.size_ = 0;
    return *this;
  }

  ~PODResizeableArray() { if (data_ != NULL) free(data_); }

  // iterators:
  iterator begin() { return iterator(&data_[0]); }
  const_iterator begin() const { return const_iterator(&data_[0]); }
  iterator end() { return iterator(&data_[size_]); }
  const_iterator end() const { return const_iterator(&data_[size_]); }

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

  // size:
  size_type size() const { return size_; }
  size_type max_size() const { return capacity_; }
  bool empty() const { return size_ == 0; }

  void reserve(size_t n) {
    if (n > capacity_) {
      if (capacity_ == 0) {
        capacity_ = 1;
      }
      while (capacity_ < n) {
        capacity_ <<= 1;
      }
      data_ = static_cast<_Tp*>(realloc(reinterpret_cast<void*>(data_), capacity_ * sizeof(_Tp)));
    }
  }

  void resize(size_t n) {
    reserve(n);
    size_ = n;
  }

  void clear() {
    size_ = 0;
  }

  // element access:
  reference operator[](size_type __n) { return data_[__n]; }
  const_reference operator[](size_type __n) const { return data_[__n]; }
  reference at(size_type __n) {
    if (__n >= size_)
      throw std::out_of_range("PODResizeableArray::at");
    return data_[__n]; 
  }
  const_reference at(size_type __n) const {
    if (__n >= size_)
      throw std::out_of_range("PODResizeableArray::at");
    return data_[__n]; 
  }

  void assign(iterator first, iterator last) {
    size_t n = last - first;
    resize(n);
    memcpy(reinterpret_cast<void*>(data_), first, n * sizeof(_Tp));
  }

  reference front() { return data_[0]; }
  const_reference front() const { return data_[0]; }
  reference back() { return data_[size_ - 1]; }
  const_reference back() const { return data_[size_ - 1]; }

  pointer data() { return data_; }
  const_pointer data() const { return data_; }

  void push_back(const _Tp& value) {
    resize(size_ + 1);
    data_[size_ - 1] = value;
  }

  template <class InputIterator>
  void insert(iterator position, InputIterator first, InputIterator last) {
    assert(position == end());
    size_t old_size = size_;
    size_t to_add = last - first;
    resize(old_size + to_add);
    std::copy_n(first, to_add, begin() + old_size);
  }

  void swap(PODResizeableArray& v) {
    std::swap(data_, v.data_);
    std::swap(size_, v.size_);
    std::swap(capacity_, v.capacity_);
  }
};

} // namespace galois
#endif // GALOIS_PODRESIZEABLEARRAY_H
