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

#ifndef EDGE_H
#define EDGE_H

#include "Tuple.h"

class Element;

class Edge {
  Tuple p[2];

public:
  Edge() {}
  Edge(const Tuple& a, const Tuple& b) {
    if (a < b) {
      p[0] = a;
      p[1] = b;
    } else {
      p[0] = b;
      p[1] = a;
    }
  }
  Edge(const Edge& rhs) {
    p[0] = rhs.p[0];
    p[1] = rhs.p[1];
  }

  bool operator==(const Edge& rhs) const {
    return p[0] == rhs.p[0] && p[1] == rhs.p[1];
  }
  bool operator!=(const Edge& rhs) const { return !(*this == rhs); }
  bool operator<(const Edge& rhs) const {
    return (p[0] < rhs.p[0]) || ((p[0] == rhs.p[0]) && (p[1] < rhs.p[1]));
  }

  bool operator>(const Edge& rhs) const {
    return (p[0] > rhs.p[0]) || ((p[0] == rhs.p[0]) && (p[1] > rhs.p[1]));
  }

  Tuple getPoint(int i) const { return p[i]; }
};
#endif
