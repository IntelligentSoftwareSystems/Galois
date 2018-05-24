/** Some debug utilities etc. -*- C++ -*-
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
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */

#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <vector>

template <typename T>
std::ostream& operator << (std::ostream& out, const std::vector<T>& v) {
  out << "{ ";
  for (typename std::vector<T>::const_iterator i = v.begin(); i != v.end(); ++i) {
    out << *i << ", ";
  }
  out << "}";

  return out;
}

template <typename I>
void printIter (std::ostream& out, I begin, I end) {
  out << "{ ";
  for (I i = begin; i != end; ++i) {
    out << *i << ", ";
  }
  out << "}" << std::endl;
}

#endif
