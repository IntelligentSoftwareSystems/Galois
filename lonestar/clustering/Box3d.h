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
