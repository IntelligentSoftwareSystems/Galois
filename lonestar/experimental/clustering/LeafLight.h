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

#ifndef LEAF_LIGHT_H
#define LEAF_LIGHT_H

#include "common.h"
#include "AbstractLight.h"
#include "Point3.h"

#include <iostream>

class LeafLight : public AbstractLight {
protected:
  // direction of maximum emission
  Point3 direction;

public:
  LeafLight(double x, double y, double z, double dirX, double dirY, double dirZ)
      : AbstractLight(x, y, z), direction(dirX, dirY, dirZ) {

    AbstractLight::setIntensity(1.0 / MATH_PI, 0);
  }

  Point3& getDirection() { return direction; }

  const Point3& getDirection() const { return direction; }

  double getDirX() const { return direction.getX(); }

  double getDirY() const { return direction.getY(); }

  double getDirZ() const { return direction.getZ(); }

  bool isLeaf() const { return true; }

  int size() const { return 1; }

  friend std::ostream& operator<<(std::ostream& s, LeafLight& pt);
};

std::ostream& operator<<(std::ostream& s, const LeafLight& pt) {
  s << "LeafLight :: ";
  operator<<(s, (AbstractLight&)pt);
  s << "Dir::" << pt.direction;
  return s;
}

#endif /* LEAF_LIGHT_H */
