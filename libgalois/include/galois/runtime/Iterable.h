/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
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

#ifndef GALOIS_RUNTIME_ITERABLE_H
#define GALOIS_RUNTIME_ITERABLE_H

namespace galois {
namespace runtime {

// iterable and make_iterable specific
// From:
// https://github.com/CppCon/CppCon2014/tree/master/Presentations/C%2B%2B11%20in%20the%20Wild%20-%20Techniques%20from%20a%20Real%20Codebase
// Author: Arthur O'Dwyer
// License: The C++ code in this directory is placed in the public domain and
// may be reused or modified for any purpose, commercial or non-commercial.

template <class It>
class iterable {
  It m_first, m_last;

public:
  iterable() = default;
  iterable(It first, It last) : m_first(first), m_last(last) {}
  It begin() const { return m_first; }
  It end() const { return m_last; }
};

template <class It>
static inline iterable<It> make_iterable(It a, It b) {
  return iterable<It>(a, b);
}

} // end namespace runtime
} // end namespace galois

#endif // GALOIS_RUNTIME_ITERABLE_H
