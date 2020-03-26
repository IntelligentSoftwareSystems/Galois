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

struct Point {
  double val[3];
  Point() { val[0] = val[1] = val[2] = 0.0; }
  // Point(double _x, double _y, double _z) : val{_x,_y,_z} {}
  Point(double _x, double _y, double _z) {
    val[0] = _x;
    val[1] = _y;
    val[2] = _z;
  }
  // explicit Point(double v) : val{v,v,v} {}
  explicit Point(double v) {
    val[0] = v;
    val[1] = v;
    val[2] = v;
  }

  double operator[](const int index) const { return val[index]; }

  double& operator[](const int index) { return val[index]; }

  double x() const { return val[0]; }

  double y() const { return val[1]; }

  double z() const { return val[2]; }

  bool operator==(const Point& other) const {
    return val[0] == other.val[0] && val[1] == other.val[1] &&
           val[2] == other.val[2];
  }

  bool operator!=(const Point& other) const { return !operator==(other); }

  Point& operator+=(const Point& other) {
    for (int i = 0; i < 3; ++i)
      val[i] += other.val[i];
    return *this;
  }

  Point& operator-=(const Point& other) {
    for (int i = 0; i < 3; ++i)
      val[i] -= other.val[i];
    return *this;
  }

  Point& operator*=(double value) {
    for (int i = 0; i < 3; ++i)
      val[i] *= value;
    return *this;
  }

  Point operator-(const Point& other) const {
    return Point(val[0] - other.val[0], val[1] - other.val[1],
                 val[2] - other.val[2]);
  }

  Point operator+(const Point& other) const {
    return Point(val[0] + other.val[0], val[1] + other.val[1],
                 val[2] + other.val[2]);
  }

  Point operator*(double d) const {
    return Point(val[0] * d, val[1] * d, val[2] * d);
  }

  Point operator/(double d) const {
    return Point(val[0] / d, val[1] / d, val[2] / d);
  }

  double dist2() const { return dot(*this); }

  double dot(const Point& p2) const {
    return val[0] * p2.val[0] + val[1] * p2.val[1] + val[2] * p2.val[2];
  }

  void pairMin(const Point& p2) {
    for (int i = 0; i < 3; ++i)
      if (p2.val[i] < val[i])
        val[i] = p2.val[i];
  }

  void pairMax(const Point& p2) {
    for (int i = 0; i < 3; ++i)
      if (p2.val[i] > val[i])
        val[i] = p2.val[i];
  }

  double minDim() const { return std::min(val[0], std::min(val[1], val[2])); }
};

std::ostream& operator<<(std::ostream& os, const Point& p) {
  os << "(" << p[0] << "," << p[1] << "," << p[2] << ")";
  return os;
}
