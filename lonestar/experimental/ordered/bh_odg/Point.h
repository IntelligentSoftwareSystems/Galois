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

#ifndef BH_POINT_H
#define BH_POINT_H

namespace bh {

bool checkRelativeError(const double ref, const double obs) {
  const double THRESHOLD = 1.0e-10;
  return fabs((ref - obs) / ref) < THRESHOLD;
}

struct Point {
  double x, y, z;
  Point() : x(0.0), y(0.0), z(0.0) {}
  Point(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
  explicit Point(double v) : x(v), y(v), z(v) {}

  double operator[](const int index) const {
    switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    }
    assert(false && "index out of bounds");
    abort();
  }

  double& operator[](const int index) {
    switch (index) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return z;
    }
    assert(false && "index out of bounds");
    abort();
  }

  bool operator==(const Point& other) {
    if (x == other.x && y == other.y && z == other.z)
      return true;
    return false;
  }

  bool operator!=(const Point& other) { return !operator==(other); }

  Point& operator+=(const Point& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }

  Point& operator-=(const Point& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }

  Point& operator*=(double value) {
    x *= value;
    y *= value;
    z *= value;
    return *this;
  }

  double mag() const { return sqrt(x * x + y * y + z * z); }

  friend std::ostream& operator<<(std::ostream& os, const Point& p) {
    os << "(" << p[0] << "," << p[1] << "," << p[2] << ")";
    return os;
  }

  friend bool checkRelativeError(const Point& ref, const Point& obs) {
    // Point tmp (ref);
    // tmp -= obs;
    // return (tmp.mag () / ref.mag ()) < THRESHOLD;
    return checkRelativeError(ref.x, obs.x) &&
           checkRelativeError(ref.y, obs.y) && checkRelativeError(ref.z, obs.z);
  }
};

} // end namespace bh
#endif // BH_POINT_H
