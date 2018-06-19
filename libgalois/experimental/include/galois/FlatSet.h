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

#ifndef GALOIS_FLAT_SET_H
#define GALOIS_FLAT_SET_H

#include "galois/gstl.h"

#include <algorithm>

namespace galois {

template <typename T, typename C = typename gstl::Vector<T>>
class FlatSet {

  C vec;

  typename C::iterator findInternal(const T& key) {
    return std::find(vec.begin(), vec.end(), key);
  }

public:
  using const_iterator = typename C::const_iterator;

  FlatSet(const C& cont = C()) : vec(cont) {}

  bool empty(void) const { return vec.empty(); }

  typename C::size_type size(void) const { return vec.size(); }

  void clear(void) { vec.clear(); }

  typename C::const_iterator find(const T& key) const {
    return std::find(vec.begin(), vec.end(), key);
  }

  typename C::size_type count(const T& key) const {
    size_t c = std::count(vec.begin(), vec.end(), key);
    assert(c <= 1);
    return c;
  }

  bool contains(const T& key) const { return find(key) != vec.end(); }

  void insert(const T& key) {
    if (find(key) == vec.end()) {
      vec.push_back(key);
    }
  }

  void erase(const T& key) {

    auto pos = findInternal(key);
    assert(pos != vec.end());

    if (pos != vec.end()) {
      auto last = vec.end();
      --last;

      std::swap(*pos, *last);
      assert(vec.back() == key);
      vec.pop_back();
    }
  }

  typename C::const_iterator cbegin() const { return vec.cbegin(); }

  typename C::const_iterator cend() const { return vec.cend(); }

  typename C::const_iterator begin() const { return cbegin(); }

  typename C::const_iterator end() const { return cend(); }
};

} // end namespace galois

#endif //  GALOIS_FLAT_SET_H
