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

#ifndef _VEC2_H_
#define _VEC2_H_

#include <string>
#include <iostream>

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <limits>
#include "FPutils.h"

class Vec2 {

protected:
  FP x;
  FP y;

public:
  Vec2(const FP& x, const FP& y) : x(x), y(y) {}

  // explicit Vec2 (FP v): x (v), y (v) {}

  FP& getX() { return x; }

  const FP& getX() const { return x; }

  FP& getY() { return y; }

  const FP& getY() const { return y; }

  FP dot(const Vec2& that) const { return ((x * that.x) + (y * that.y)); }

  FP magSqrd() const {
    double d = double(this->dot(*this));
    assert(d >= 0.0);

    return this->dot(*this);
  }

  FP mag() const { return FPutils::sqrt(magSqrd()); }

  Vec2 unit() const {

    Vec2 v(*this);
    v /= v.mag();
    return v;
  }

  //! the left normal computed by
  //! swapping x and y components and
  //! negating x componen of the result
  Vec2 leftNormal() const { return Vec2(-y, x); }

  //! the right normal computed by
  //! swapping x and y components and
  //! negating y component of the result
  Vec2 rightNormal() const { return Vec2(y, -x); }

  Vec2& operator*=(const FP& s) {

    x *= s;
    y *= s;

    return *this;
  }

  Vec2& operator/=(const FP& s) {

    x /= s;
    y /= s;

    return *this;
  }

  Vec2& operator+=(const Vec2& that) {

    x += that.x;
    y += that.y;

    return *this;
  }

  Vec2& operator-=(const Vec2& that) {

    x -= that.x;
    y -= that.y;

    return *this;
  }

  FP dist(const Vec2& that) const {
    Vec2 t(*this);
    t -= that;
    return t.mag();
  }

  const std::string str() const {

    char s[256];
    sprintf(s, "(%10.10f,%10.10f)", double(x), double(y));

    return s;
  }
};

Vec2 operator+(const Vec2& v1, const Vec2& v2);

Vec2 operator-(const Vec2& v1, const Vec2& v2);

Vec2 operator/(const Vec2& v1, const FP& s);

Vec2 operator*(const Vec2& v1, const FP& s);

Vec2 operator*(const FP& s, const Vec2& v1);

std::ostream& operator<<(std::ostream& out, const Vec2& v);

class BoundingBox {

  Vec2 m_min;
  Vec2 m_max;

public:
  using D = typename std::numeric_limits<FP>;

  BoundingBox(void)
      : m_min(D::max(), D::max()), m_max(D::min(), D::min())

  {}

  std::string str(void) const {
    char s[256];

    std::sprintf(s, "BB:<%s,%s>", m_min.str().c_str(), m_max.str().c_str());

    return s;
  }

  void update(const Vec2& p) {
    if (p.getX() < m_min.getX()) {
      m_min.getX() = p.getX();
    }

    if (p.getY() < m_min.getY()) {
      m_min.getY() = p.getY();
    }

    if (p.getX() > m_max.getX()) {
      m_max.getX() = p.getX();
    }

    if (p.getY() > m_max.getY()) {
      m_max.getY() = p.getY();
    }
  }

  const Vec2& getMin(void) const { return m_min; }

  const Vec2& getMax(void) const { return m_max; }

  bool isInside(const Vec2& p) const {
    if (p.getX() >= m_min.getX() && p.getY() >= m_min.getY() &&
        p.getX() <= m_max.getX() && p.getY() <= m_max.getY()) {

      return true;

    } else {
      return false;
    }
  }
};

class LineSegment {

  Vec2 pt1;
  Vec2 pt2;

  FP detlaY(void) const { return pt2.getY() - pt1.getY(); }

  FP deltaX(void) const { return pt2.getX() - pt1.getX(); }

public:
  LineSegment(const Vec2& pt1, const Vec2& pt2) : pt1(pt1), pt2(pt2) {}

  std::string str() const {

    char s[256];
    sprintf(s, "[LineSegment: start: %s, end: %s]", pt1.str().c_str(),
            pt2.str().c_str());

    return s;
  }

  const Vec2& getBegin(void) const { return pt1; }

  const Vec2& getEnd(void) const { return pt2; }

  Vec2 lengthVec(void) const { return (pt2 - pt1); }

  FP distanceFrom(const Vec2& p) const {

    // consider the vector from pt1 to p, call it V
    // and the lengthVec, which is from pt1 to pt2 , call it L
    // if V's projection on to L is > V or < 0, then point is outside
    // the line-segment and we measure distance from the endpoints
    // else distance is V - V's projection on to L
    Vec2 L            = lengthVec();
    const FP Lsquared = L.magSqrd();

    if (Lsquared == FP(0.0)) {
      return (p - pt1).mag();

    } else {
      Vec2 V = p - pt1;

      // V's projection on to L, normalized by |L|
      const FP r = (L.dot(V)) / Lsquared;

      if (r < FP(0.0)) { // p lies to the left of pt1
        return (p - pt1).mag();

      } else if (r > FP(1.0)) { // p lies to the right of pt2
        return (p - pt2).mag();

      } else { // p lies between pt1 and pt2

        Vec2 proj = pt1 + r * L;
        return (p - proj).mag();
      }
    }
  }

}; // end LineSegment

enum class RectSide : int {
  // clockwise numbering
  LEFT   = 0,
  TOP    = 1,
  RIGHT  = 2,
  BOTTOM = 3
};

#endif // _VEC2_H_
