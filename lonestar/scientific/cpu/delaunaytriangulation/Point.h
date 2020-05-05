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

#ifndef POINT_H
#define POINT_H

#include "Tuple.h"
#include "Graph.h"

#include "galois/CheckedObject.h"

#include <ostream>
#include <algorithm>

class Point : public galois::GChecked<void> {
  Tuple m_t;
  GNode m_n;
  long m_id;

public:
  Point(double x, double y, long id) : m_t(x, y), m_n(NULL), m_id(id) {}

  const Tuple& t() const { return m_t; }
  long id() const { return m_id; }

  Tuple& t() { return m_t; }
  long& id() { return m_id; }

  void addElement(const GNode& n) { m_n = n; }

  void removeElement(const GNode& n) {
    if (m_n == n)
      m_n = NULL;
  }

  bool inMesh() const { return m_n != NULL; }

  GNode someElement() const { return m_n; }

  void print(std::ostream& os) const {
    os << "(id: " << m_id << " t: ";
    m_t.print(os);
    if (m_n != NULL)
      os << " SOME)";
    else
      os << " NULL)";
  }
};

static inline std::ostream& operator<<(std::ostream& os, const Point& rhs) {
  rhs.print(os);
  return os;
}

#endif
