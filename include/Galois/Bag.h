/** Bags -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_BAG_H
#define GALOIS_BAG_H

#include "Galois/Runtime/InsBag.h"

#include <boost/iterator/iterator_facade.hpp>
#include <list>
#include <vector>

namespace Galois {

/**
 * Bag for only concurrent insertions. This data structure
 * supports scalable concurrent pushes but reading the bag
 * can only be done serially.
 */
template<typename T>
class InsertBag: public GaloisRuntime::galois_insert_bag<T> {

};

/**
 * Bag for sequential use.
 */
// XXX: when used for reduction may have memory contention problems?
// ie. fix allocator for vector?
template<typename T>
class Bag {
  typedef std::vector<T> Inner;
  typedef std::list<Inner> Outer;

  Outer m_data;
  static const size_t capacity = 1024;

  Inner& currentInner() {
    if (!m_data.empty() && m_data.back().size() < capacity) {
      return m_data.back();
    } else {
      Inner x;
      x.reserve(capacity);
      m_data.push_back(x);
      return m_data.back();
    }
  }

  template<typename Value, typename OuterRef, typename OuterIterator, typename InnerIterator>
  class MyIterator : public boost::iterator_facade<MyIterator<Value,OuterRef,OuterIterator,InnerIterator>, Value, boost::forward_traversal_tag> {
    OuterIterator m_outer;
    InnerIterator m_inner_current, m_inner_end;

    MyIterator(OuterRef x) {
      m_outer = x.begin();
      if (!x.empty()) {
        m_inner_current = x.front().begin();
        m_inner_end = x.front().end();
      }
    }

    MyIterator(OuterRef x, bool end) {
      m_outer = x.end();
    }

    friend class boost::iterator_core_access;
    friend class Bag;

    bool equal(MyIterator<Value,OuterRef,OuterIterator,InnerIterator> const &other) const {
      return m_outer == other.m_outer;
    }
    
    void increment() {
      ++m_inner_current;
      if (m_inner_current == m_inner_end) {
        ++m_outer;
        m_inner_current = m_outer->begin();
        m_inner_end - m_outer->end();
      }
    }
    
    Value& dereference() const {
      return *m_inner_current;
    }
  };

public:
  typedef MyIterator<T, Outer&, typename Outer::iterator, typename Inner::iterator> iterator;
  typedef MyIterator<T const, const Outer&, typename Outer::const_iterator, typename Inner::const_iterator> const_iterator;

  struct Merge : public std::binary_function<Bag<T>,Bag<T>,void> {
    void operator()(Bag& a, Bag& b) {
      Outer& x = a.m_data;
      x.splice(x.begin(), b.m_data);
    }
  };

  const_iterator begin() const {
    return const_iterator(m_data);
  }

  const_iterator end() const {
    return const_iterator(m_data, true);
  }

  iterator begin() {
    return iterator(m_data);
  }

  iterator end() {
    return iterator(m_data, true);
  }

  void push(const T& x) {
    currentInner().push_back(x);
  }
};

}
#endif
