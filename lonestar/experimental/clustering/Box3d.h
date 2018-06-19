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

#ifndef BOX3D_H_
#define BOX3D_H_

#include "Point3.h"
#include <limits>

class Box3d {
protected:
  Point3 m_min;
  Point3 m_max;
  bool m_init;

public:
  Box3d()
      : m_min(std::numeric_limits<double>::m_max()),
        m_max(-1 * std::numeric_limits<double>::m_max()), m_init(false) {}

  explicit Box3d(const Point3& pt)
      : m_min(pt), m_max(pt), m_init(true)

  {}

  Box3d(const Point3& a, const Point3& b) : m_min(a), m_max(a), m_init(true) {
    addPoint(b);
  }

  void addPoint(const Point3& pt) {
    m_init = true;
    m_min.setIfMin(pt);
    m_max.setIfMax(pt);
  }

  void addBox(const Box3d& b) {
    m_init = true;
    m_min.setIfMin(b.m_min);
    m_max.setIfMax(b.m_max);
  }

  Point3 size(void) const {

    Point3 ret(m_max);
    ret.absDiff(m_min);

    return ret;
  }

  const Point3& getMin() const { return m_min; }

  const Point3& getMax() const { return m_max; }

  bool isInitialized() const { return m_init; }

  bool operator==(const Box3d& other) const {
    return m_min == other.m_min && m_max == other.m_max;
  }
};

#endif /* BOX3D_H_ */
