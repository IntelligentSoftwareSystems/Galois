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

#ifndef _ED_H_
#define _ED_H_

#include "BCNode.h"
#include "control.h"

struct BCEdge {
  using NodeType = BCNode<BC_USE_MARKING, BC_CONCURRENT>;
  NodeType* src;
  NodeType* dst;
  ShortPathType val;
  unsigned level;

  BCEdge(NodeType* _src, NodeType* _dst)
      : src(_src), dst(_dst), val(0), level(infinity) {}
  BCEdge() : src(0), dst(0), val(0), level(infinity) {}

  BCEdge& operator=(BCEdge const& from) {
    if (this != &from) {
      src   = from.src;
      dst   = from.dst;
      val   = from.val;
      level = from.level;
    }
    return *this;
  }

  inline void reset() {
    if (level != infinity) {
      level = infinity;
    }
  }

  void checkClear(int j) {
    if (level != infinity) {
      galois::gError(j, " PROBLEM WITH LEVEL OF ", toString());
    }
    if (val != 0) {
      galois::gError(j, " PROBLEM WITH VAL OF ", toString());
    }
  }

  /**
   * TODO actually implement this if needed
   */
  // char isAlreadyIn() {
  //  return 0;
  //}

  std::string toString() const {
    std::ostringstream s;
    s << val << " " << level;
    return s.str();
  }
};
#endif
