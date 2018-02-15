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

#ifndef LEAFNODE_H_
#define LEAFNODE_H_

#include "AbstractNode.h"
#include "Point3.h"

#include <iostream>

#include <cmath>

#define MATH_PI 3.1415926

class LeafNode : public AbstractNode {
protected:

  constexpr static const double MATH_PI = std::acos(-1);

  // direction of maximum emission
  Point3 direction;

public:

  LeafNode(double x, double y, double z, double dirX, double dirY, double dirZ)
      : AbstractNode(x, y, z), direction(dirX, dirY, dirZ) {

        AbstractNode::setIntensity(1.0 / MATH_PI, 0);
  }

  Point3& getDirection() { return direction; }

  const Point3& getDirection() const { return direction; }

  double getDirX() const { return direction.getX(); }

  double getDirY() const { return direction.getY(); }

  double getDirZ() const { return direction.getZ(); }

  bool isLeaf() const { return true; }

  int size() const { return 1; }

  friend std::ostream& operator<<(std::ostream& s, LeafNode& pt);
};

std::ostream& operator<<(std::ostream& s, const LeafNode& pt) {
  s << "LeafNode :: ";
  operator<<(s, (AbstractNode&)pt);
  s << "Dir::" << pt.direction;
  return s;
}

#endif /* LEAFNODE_H_ */
