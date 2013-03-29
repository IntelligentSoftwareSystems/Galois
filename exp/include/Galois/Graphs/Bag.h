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

#include "Galois/Runtime/PerHostStorage.h"

namespace Galois {
namespace Graph {

template<typename T>
class Bag :boost::noncopyable {

  typedef std::list<T, Runtime::MM::FSBGaloisAllocator<T>> ConTy;

  Runtime::PerThreadStorage<ConTy> items;
  
  Runtime::PerHost<Bag> basePtr;

public:
  typedef Runtime::PerHost<Bag> pointer;
  static pointer allocate() {
    return Runtime::PerHost<Bag>::allocate();
  }
  static void deallocate(pointer ptr) {
    Runtime::PerHost<Bag>::deallocate(ptr);
  }

  explicit Bag(pointer p) :basePtr(p) {}
  Bag(pointer p, Runtime::Distributed::DeSerializeBuffer&) :basePtr(p) {}

  void getInitData(Runtime::Distributed::SerializeBuffer&) {}

  //LOCAL operations

  typedef typename ConTy::iterator local_iterator;
  typedef typename ConTy::const_iterator local_const_iterator;

  local_iterator local_begin() { return items.getLocal()->begin(); }
  local_iterator local_end() { return items.getLocal()->end(); }
  local_const_iterator local_cbegin() { return items.getLocal()->cbegin(); }
  local_const_iterator local_cend() { return items.getLocal()->cend(); }

  template<typename... Args>
  local_iterator emplace(Args&&... args) {
    items.getLocal()->emplace_front(std::forward<Args>(args)...);
    return items.getLocal()->begin();
  }

  //! Thread safe bag insertion
  void push(const T& val) { items.getLocal()->push_front(val); }
  //! Thread safe bag insertion
  void push(T&& val) { items.getLocal()->push_front(std::move(val)); }

  //! Thread safe bag insertion
  void push_back(const T& val) { items.getLocal()->push_front(val); }
  //! Thread safe bag insertion
  void push_back(T&& val) { items.getLocal()->push_front(std::move(val)); }

  //REMOTE Aware operations

  class iterator :public std::iterator<std::forward_iterator_tag, gptr<T>> {
  };

  iterator begin();
  iterator end();
 
};

} // namespace Graph
} // namespace Galois

#endif
