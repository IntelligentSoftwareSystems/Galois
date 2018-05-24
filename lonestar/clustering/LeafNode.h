#ifndef LEAFNODE_H_
#define LEAFNODE_H_

#include "AbstractNode.h"
#include "Point3.h"
#include <iostream>

#define MATH_PI 3.1415926

class LeafNode : public AbstractNode {
protected:
  // direction of maximum emission
  Point3 direction;
  /**
   * Creates a new instance of MLTreeLeafNode
   */
public:
  LeafNode(double x, double y, double z, double dirX, double dirY, double dirZ)
      : AbstractNode(x, y, z), direction(dirX, dirY, dirZ) {
    setIntensity(1.0 / MATH_PI, 0);
  }

  Point3& getDirection() { return direction; }
  double getDirX() { return direction.getX(); }
  double getDirY() { return direction.getY(); }
  double getDirZ() { return direction.getZ(); }
  bool isLeaf() { return true; }

  int size() { return 1; }
  friend std::ostream& operator<<(std::ostream& s, LeafNode& pt);
};

std::ostream& operator<<(std::ostream& s, LeafNode& pt) {
  s << "LeafNode :: ";
  operator<<(s, (AbstractNode&)pt);
  s << "Dir::" << pt.direction;
  return s;
}

#endif /* LEAFNODE_H_ */
