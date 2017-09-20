/** Bags -*- C++ -*-
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
 * Large unordered collections of things.
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_GRAPH_BAG_H
#define GALOIS_GRAPH_BAG_H

#include "Galois/Runtime/Serialize.h"
#include "Galois/Runtime/PerHostStorage.h"
#include "Galois/TwoLevelIteratorA.h"

namespace galois {
namespace graphs {

template<typename T>
class Bag :boost::noncopyable, public runtime::Lockable {

  typedef gdeque<T> ConTy;
  ConTy items;

  runtime::PerThreadDist<Bag> basePtr;

public:
  typedef runtime::PerThreadDist<Bag> pointer;
  static pointer allocate() {
    return runtime::PerThreadDist<Bag>::allocate();
  }
  static void deallocate(pointer ptr) {
    runtime::PerThreadDist<Bag>::deallocate(ptr);
  }

  Bag() {}
  explicit Bag(pointer p) :basePtr(p) {}
  Bag(pointer p, runtime::DeSerializeBuffer&) :basePtr(p) {}
  Bag(runtime::DeSerializeBuffer& buf) { deserialize(buf); }

  void getInitData(runtime::SerializeBuffer&) {}

  //LOCAL operations

  typedef typename ConTy::iterator local_iterator;

  local_iterator local_begin() { return items.begin(); }
  local_iterator local_end() { return items.end(); }

  template<typename... Args>
  local_iterator emplace(Args&&... args) {
    items.emplace_front(std::forward<Args>(args)...);
    return local_begin();
  }

  //! Thread safe bag insertion
  void push(const T& val) { items.push_front(val); }
  //! Thread safe bag insertion
  void push(T&& val) { items.push_front(std::move(val)); }

  //! Thread safe bag insertion
  void push_back(const T& val) { items.push_front(val); }
  //! Thread safe bag insertion
  void push_back(T&& val) { items.push_front(std::move(val)); }

  T& back() { return items.front(); }

  struct InnerBegFnL : std::unary_function<runtime::gptr<Bag>, local_iterator> {
    local_iterator operator()(runtime::gptr<Bag> d) {
      acquire(d, MethodFlag::ALL);
      return d->local_begin();
    }
  };
  struct InnerEndFnL : std::unary_function<runtime::gptr<Bag>, local_iterator> {
    local_iterator operator()(runtime::gptr<Bag> d) {
      acquire(d, MethodFlag::ALL);
      return d->local_end();
    }
  };

  typedef TwoLevelIteratorA<typename runtime::PerThreadDist<Bag>::iterator, local_iterator, std::forward_iterator_tag, InnerBegFnL, InnerEndFnL> iterator;
  iterator begin() { return iterator(basePtr.begin(), basePtr.end(), basePtr.begin(), InnerBegFnL(), InnerEndFnL()); }
  iterator end() { return iterator(basePtr.end(), basePtr.end(), basePtr.end(), InnerBegFnL(), InnerEndFnL()); }

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(galois::runtime::SerializeBuffer& s) const {
    gSerialize(s,basePtr, items);
  }
  void deserialize(galois::runtime::DeSerializeBuffer& s) {
    gDeserialize(s,basePtr, items);
  }
};

} // namespace graphs
} // namespace galois

#endif
