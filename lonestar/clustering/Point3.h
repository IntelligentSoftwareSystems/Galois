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

#ifndef POINT3_H_
#define POINT3_H_

#include <iostream>

class Point3 {

  double x, y, z;

public:
  Point3(double v) { this->set(v); }

  Point3(double x, double y, double z) {
    this->x = x;
    this->y = y;
    this->z = z;
  }

  Point3(const Point3& pt) {
    this->x = pt.x;
    this->y = pt.y;
    this->z = pt.z;
  }
  double getSum() const { return x + y + z; }

  double getLen() const { return x * x + y * y + z * z; }

  void scale(double factor) {
    x *= factor;
    y *= factor;
    z *= factor;
  }

  void add(const Point3& pt) {
    x += pt.x;
    y += pt.y;
    z += pt.z;
  }

  void sub(const Point3& pt) {
    x -= pt.x;
    y -= pt.y;
    z -= pt.z;
  }

  void set(double n) { x = y = z = n; }

  void set(double x, double y, double z) {
    this->x = x;
    this->y = y;
    this->z = z;
  }

  void set(const Point3& other) {
    x = other.x;
    y = other.y;
    z = other.z;
  }

  bool setIfMax(double nx, double ny, double nz) {
    bool ret = false;
    if (nx > x) {
      x   = nx;
      ret = true;
    }
    if (ny > y) {
      y   = ny;
      ret = true;
    }
    if (nz > z) {
      z   = nz;
      ret = true;
    }
    return ret;
  }

  bool setIfMin(double nx, double ny, double nz) {
    bool ret = false;
    if (nx < x) {
      x   = nx;
      ret = true;
    }
    if (ny < y) {
      y   = ny;
      ret = true;
    }
    if (nz < z) {
      z   = nz;
      ret = true;
    }
    return ret;
  }

  bool setIfMax(const Point3& other) {
    return setIfMax(other.x, other.y, other.z);
  }

  bool setIfMin(const Point3& other) {
    return setIfMin(other.x, other.y, other.z);
  }

  double getX() const { return x; }

  double getY() const { return y; }
  
  double getZ() const { return z; }

  bool equals(const Point3& other) const {
    return (x == other.x) && (y == other.y) && (z == other.z);
  }

  friend std::ostream& operator<<(std::ostream& s, const Point3& pt);
};

std::ostream& operator<<(std::ostream& s, const Point3& pt) {
  s << "(" << pt.x << "," << pt.y << "," << pt.z << ")";
  return s;
}

#endif /* POINT3_H_ */
