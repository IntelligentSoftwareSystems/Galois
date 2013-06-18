/** Fixed-size ring buffer -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_FIXEDSIZERING_H
#define GALOIS_FIXEDSIZERING_H

#include "Galois/config.h"
#include "Galois/optional.h"
#include "Galois/LazyArray.h"

#include <boost/iterator/iterator_facade.hpp>
#include <boost/utility.hpp>

#include GALOIS_CXX11_STD_HEADER(utility)

namespace Galois {

//! Unordered collection of bounded size
template<typename T, unsigned chunksize = 64>
class FixedSizeBag: private boost::noncopyable {
  LazyArray<T, chunksize> datac;
  unsigned count;

  T* at(unsigned i) { return &datac[i]; }
  const T* at(unsigned i) const { return &datac[i]; }

  bool precondition() const {
    return count <= chunksize;
  }

public:
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef pointer iterator;
  typedef const_pointer const_iterator;

  FixedSizeBag(): count(0) { }

  ~FixedSizeBag() {
    clear();
  }

  unsigned size() const {
    assert(precondition());
    return count;
  }

  bool empty() const {
    assert(precondition());
    return count == 0;
  }

  bool full() const {
    assert(precondition());
    return count == chunksize;
  }

  void clear() {
    assert(precondition());
    for (unsigned x = 0; x < count; ++x)
      datac.destroy(x);
    count = 0;
  }

  template<typename U>
  pointer push_front(U&& val) { return push_back(std::forward<U>(val)); }

  template<typename... Args>
  pointer emplace_front(Args&&... args) { return emplace_back(std::forward<Args>(args)...); }

  template<typename U>
  pointer push_back(U&& val) {
    if (full()) return 0;
    unsigned end = count;
    ++count;
    return datac.construct(end, std::forward<U>(val));
  }

  template<typename... Args>
  pointer emplace_back(Args&&... args) {
    if (full()) return 0;
    unsigned end = count;
    ++count;
    return datac.emplace(end, std::forward<Args>(args)...);
  }

  reference front() { return back(); }
  const_reference front() const { return back(); }
  Galois::optional<value_type> extract_front() { return extract_back(); }

  void pop_front() {
    pop_back();
  }
  
  reference back() {
    assert(precondition());
    assert(!empty());
    return *at(count - 1);
  }

  const_reference back() const {
    assert(precondition());
    assert(!empty());
    return *at(count - 1);
  }

  Galois::optional<value_type> extract_back() {
    if (!empty()) {
      Galois::optional<value_type> retval(back());
      pop_back();
      return retval;
    }
    return Galois::optional<value_type>();
  }

  void pop_back() {
    assert(precondition());
    assert(!empty());
    unsigned end = (count - 1);
    datac.destroy(end);
    --count;
  }

  iterator begin() { return &datac[0]; }
  iterator end() { return &datac[count]; }
  const_iterator begin() const { return &datac[0]; }
  const_iterator end() const { return &datac[count]; }
};
 
//! Ordered collection of bounded size
template<typename T, unsigned chunksize = 64>
class FixedSizeRing: private boost::noncopyable {
  LazyArray<T, chunksize> datac;
  unsigned start;
  unsigned count;

  T* at(unsigned i) { return &datac[i]; }
  const T* at(unsigned i) const { return &datac[i]; }

  bool precondition() const {
    return count <= chunksize && start <= chunksize;
  }

  template<typename U, bool isForward>
  class Iterator: public boost::iterator_facade<Iterator<U,isForward>, U, boost::forward_traversal_tag> {
    friend class boost::iterator_core_access;
    U* base;
    unsigned cur;
    unsigned count;

    template<typename OtherTy, bool OtherIsForward>
    bool equal(const Iterator<OtherTy, OtherIsForward>& o) const {
      return base + cur == o.base + o.cur && count == o.count;
    }

    U& dereference() const { return base[cur]; }

    void increment() {
      if (--count == 0) {
        base = 0; cur = 0;
      } else {
        cur = isForward ? (cur + 1) % chunksize : (cur + chunksize - 1) % chunksize;
      }
    }

  public:
    Iterator(): base(0), cur(0), count(0) { }
    
    template<typename OtherTy, bool OtherIsForward>
    Iterator(const Iterator<OtherTy, OtherIsForward>& o): base(o.base), cur(o.cur), count(o.count) { }
    
    Iterator(U* b, unsigned c, unsigned co): base(b), cur(c), count(co) { 
      if (count == 0) {
        base = 0;
        cur = 0;
      }
    }
  };

public:
  typedef T value_type;
  typedef T* pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef Iterator<T, true> iterator;
  typedef Iterator<const T, true> const_iterator;
  typedef Iterator<T, false> reverse_iterator;
  typedef Iterator<const T, false> const_reverse_iterator;

  FixedSizeRing(): start(0), count(0) { }

  ~FixedSizeRing() {
    clear();
  }

  unsigned size() const {
    assert(precondition());
    return count;
  }

  bool empty() const {
    assert(precondition());
    return count == 0;
  }

  bool full() const {
    assert(precondition());
    return count == chunksize;
  }

  reference getAt(unsigned x) {
    assert(precondition());
    assert(!empty());
    return *at((start + x) % chunksize);
  }

  const_reference getAt(unsigned x) const {
    assert(precondition());
    assert(!empty());
    return *at((start + x) % chunksize);
  }

  void clear() {
    assert(precondition());
    for (unsigned x = 0; x < count; ++x)
      datac.destroy((start + x) % chunksize);
    count = 0;
    start = 0;
  }

  template<typename U>
  pointer push_front(U&& val) {
    if (full()) return 0;
    start = (start + chunksize - 1) % chunksize;
    ++count;
    return datac.construct(start, std::forward<U>(val));
  }

  template<typename... Args>
  pointer emplace_front(Args&&... args) {
    if (full()) return 0;
    start = (start + chunksize - 1) % chunksize;
    ++count;
    return datac.emplace(start, std::forward<Args>(args)...);
  }

  template<typename U>
  pointer push_back(U&& val) {
    if (full()) return 0;
    unsigned end = (start + count) % chunksize;
    ++count;
    return datac.construct(end, std::forward<U>(val));
  }

  template<typename... Args>
  pointer emplace_back(Args&&... args) {
    if (full()) return 0;
    unsigned end = (start + count) % chunksize;
    ++count;
    return datac.emplace(end, std::forward<Args>(args)...);
  }

  reference front() {
    assert(precondition());
    assert(!empty());
    return *at(start);
  }

  const_reference front() const {
    assert(precondition());
    assert(!empty());
    return *at(start);
  }

  Galois::optional<value_type> extract_front() {
    if (!empty()) {
      Galois::optional<value_type> retval(front());
      pop_front();
      return retval;
    }
    return Galois::optional<value_type>();
  }

  void pop_front() {
    assert(precondition());
    assert(!empty());
    datac.destroy(start);
    start = (start + 1) % chunksize;
    --count;
  }
  
  reference back() {
    assert(precondition());
    assert(!empty());
    return *at((start + count - 1) % chunksize);
  }

  const_reference back() const {
    assert(precondition());
    assert(!empty());
    return *at((start + count - 1) % chunksize); 
  }

  Galois::optional<value_type> extract_back() {
    if (!empty()) {
      Galois::optional<value_type> retval(back());
      pop_back();
      return retval;
    }
    return Galois::optional<value_type>();
  }

  void pop_back() {
    assert(precondition());
    assert(!empty());
    unsigned end = (start + count - 1) % chunksize;
    datac.destroy(end);
    --count;
  }

  iterator begin() { return iterator(&datac[0], start, count); }
  iterator end() { return iterator(); }
  const_iterator begin() const { return const_iterator(&datac[0], start, count); }
  const_iterator end() const { return const_iterator(); }

  reverse_iterator rbegin() { return reverse_iterator(&datac[0], (start + count - 1) % chunksize, count); }
  reverse_iterator rend() { return reverse_iterator(); }
  const_iterator rbegin() const { const_reverse_iterator(&datac[0], (start + count - 1) % chunksize, count); }
  const_iterator rend() const { const_reverse_iterator(); }
};
 
}

#endif
