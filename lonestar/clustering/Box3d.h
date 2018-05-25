/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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
  Point3 min;
  Point3 max;
  bool initialized;

public:
  Box3d()
      : min(std::numeric_limits<float>::max()),
        max(-1 * std::numeric_limits<double>::max()) {
    initialized = false;
  }
  void setBox(Point3& pt) {
    initialized = true;
    min.set(pt);
    max.set(pt);
  }
  void addPoint(Point3& pt) {
    initialized = true;
    min.setIfMin(pt);
    max.setIfMax(pt);
  }
  void addBox(Box3d& b) {
    initialized = true;
    min.setIfMin(b.min);
    max.setIfMax(b.max);
  }
  const Point3& getMin() const { return min; }
  const Point3& getMax() const { return max; }
  bool isInitialized() const { return initialized; }
  bool equals(const Box3d& other) const {
    return min.equals(other.min) && max.equals(other.max);
  }
};

#endif /* BOX3D_H_ */
