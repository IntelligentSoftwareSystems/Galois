#ifndef BH_BOUNDING_BOX_H
#define BH_BOUNDING_BOX_H

#include "Point.h"
#include "Octree.h"

#include "Galois/Accumulator.h"

namespace bh {

struct BoundingBox {
  Point min;
  Point max;
  explicit BoundingBox(const Point& p) : min(p), max(p) { }
  BoundingBox() :
    min(std::numeric_limits<double>::max()),
    max(std::numeric_limits<double>::min()) { }

  void merge(const BoundingBox& other) {
    for (int i = 0; i < 3; i++) {
      if (other.min[i] < min[i])
        min[i] = other.min[i];
    }
    for (int i = 0; i < 3; i++) {
      if (other.max[i] > max[i])
        max[i] = other.max[i];
    }
  }

  void merge(const Point& other) {
    for (int i = 0; i < 3; i++) {
      if (other[i] < min[i])
        min[i] = other[i];
    }
    for (int i = 0; i < 3; i++) {
      if (other[i] > max[i])
        max[i] = other[i];
    }
  }

  double diameter() const {
    double diameter = max.x - min.x;
    for (int i = 1; i < 3; i++) {
      double t = max[i] - min[i];
      if (diameter < t)
        diameter = t;
    }
    return diameter;
  }

  double radius() const {
    return diameter() / 2;
  }

  Point center() const {
    return Point(
        (max.x + min.x) * 0.5,
        (max.y + min.y) * 0.5,
        (max.z + min.z) * 0.5);
  }
};

std::ostream& operator<<(std::ostream& os, const BoundingBox& b) {
  os << "(min:" << b.min << " max:" << b.max << ")";
  return os;
}

struct BoxMergeFunc {
  void operator () (BoundingBox& left, const BoundingBox& right) const {
    left.merge (right);
  }
};

struct ReducibleBox: public galois::GReducible<BoundingBox, BoxMergeFunc> {
};

struct ReduceBoxes {
  // NB: only correct when run sequentially or tree-like reduction
  typedef int tt_does_not_need_stats;
  ReducibleBox& result;

  ReduceBoxes(ReducibleBox& _result): result(_result) { }

  void operator()(const Point& p) const {
    result.update (BoundingBox (p));
  }

  template<typename Context>
  void operator()(const Point& p, Context&) const {
    operator() (p);
  }
};

} // end namespace bh

#endif //  BH_BOUNDING_BOX_H
