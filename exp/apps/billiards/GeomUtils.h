/** Basic geometry facilities  -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * Basic geometry facilities .
 *
 * @author <ahassaan@ices.utexas.edu>
 */



#ifndef _VEC2_H_
#define _VEC2_H_

#include <string>
#include <iostream>

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <limits>

class Vec2 {

protected:

  double x;
  double y;

public:

  Vec2 (double x, double y): x (x), y (y) {}

  // explicit Vec2 (double v): x (v), y (v) {}

  double& getX () { return x; }

  const double& getX () const { return x; }

  double& getY () { return y; }

  const double& getY () const { return y; }

  double dot (const Vec2& that) const {
    return ((x * that.x) + (y * that.y));
  }

  double magSqrd () const {
    return this->dot (*this);
  }

  double  mag () const {
    return sqrt (magSqrd ());
  }


  Vec2 unit () const {

    Vec2 v(*this);
    v /= v.mag ();
    return v;

  }

  //! the left normal computed by
  //! swapping x and y components and 
  //! negating x componen of the result
  Vec2 leftNormal () const { return Vec2 (-y, x); }

  //! the right normal computed by
  //! swapping x and y components and 
  //! negating y component of the result
  Vec2 rightNormal () const { return Vec2 (y, -x); }

  Vec2& operator *= (double s) {

    x *= s;
    y *= s;

    return *this;
  }

  Vec2& operator /= (double s) {

    x /= s;
    y /= s;
    
    return *this;
  }


  Vec2& operator += (const Vec2& that) {

    x += that.x;
    y += that.y;

    return *this;
  }

  Vec2& operator -= (const Vec2& that) {

    x -= that.x;
    y -= that.y;

    return *this;
  }

  double dist (const Vec2& that) const {
    Vec2 t (*this);
    t -= that;
    return t.mag ();
  }

  const std::string str () const {

    char s [256];
    sprintf (s, "(%10.10f,%10.10f)", x, y);

    return s;
  }

};


Vec2 operator + (const Vec2& v1, const Vec2& v2);

Vec2 operator - (const Vec2& v1, const Vec2& v2);

Vec2 operator / (const Vec2& v1, const double s);

Vec2 operator * (const Vec2& v1, const double s);

Vec2 operator * (const double s, const Vec2& v1);

std::ostream& operator << (std::ostream& out, const Vec2& v);

class BoundingBox {

  Vec2 m_min;
  Vec2 m_max;


public:
  using D = typename std::numeric_limits<double>;

  BoundingBox (void): 
    m_min (D::min (), D::min ()), 
    m_max (D::max (), D::max ())

  {}

  void update (const Vec2& p) {
    if (m_min.getX () < p.getX ()) {
      m_min.getX () = p.getX ();
    }

    if (m_min.getY () < p.getY ()) {
      m_min.getY () = p.getY ();
    }

    if (m_max.getX () > p.getX ()) {
      m_max.getX () = p.getX ();
    }

    if (m_max.getY () > p.getY ()) {
      m_max.getY () = p.getY ();
    }
  }

  const Vec2& getMin (void) const {
    return m_min;
  }

  const Vec2& getMax (void) const {
    return m_max;
  }

  bool isInside (const Vec2& p) const {
    if ( p.getX () >= m_min.getX ()
      && p.getY () >= m_min.getY ()
      && p.getX () <= m_max.getX ()
      && p.getY () <= m_max.getY ()) {

      return true;

    } else {
      return false;
    }
  }

};


class LineSegment {

  Vec2 pt1;
  Vec2 pt2;

  double detlaY (void) const {
    return pt2.getY () - pt1.getY ();
  }

  double deltaX (void) const {
    return pt2.getX () - pt1.getX ();
  }

public:

  LineSegment (const Vec2& pt1, const Vec2& pt2) 
    : pt1 (pt1), pt2 (pt2) 
  {}

  std::string str () const {

    char s[256];
    sprintf (s, "[LineSegment: start: %s, end: %s]", pt1, pt2);

    return s;
  }

  const Vec2& getBegin (void) const { 
    return pt1;
  }

  const Vec2& getEnd (void) const {
    return pt2;
  }

  Vec2 lengthVec (void) const { 
    return (pt2 - pt1);
  }


  double distanceFrom (const Vec2& p) const {

    // consider the vector from pt1 to p, call it V
    // and the lengthVec, which is from pt1 to pt2 , call it L
    // if V's projection on to L is > V or < 0, then point is outside 
    // the line-segment and we measure distance from the endpoints
    // else distance is V - V's projection on to L
    Vec2 L = lengthVec ();
    const double Lsquared = L.magSqrd ();

    if (Lsquared == 0.0) { 
      return (p - pt1).mag ();

    } else {
      Vec2 V = p - pt1;

      // V's projection on to L, normalized by |L|
      const double r = (L.dot (V)) / Lsquared;

      if (r < 0.0) {  // p lies to the left of pt1
        return (p - pt1).mag ();

      } else if (r > 1.0) { // p lies to the right of pt2
        return (p - pt2).mag ();

      } else { // p lies between pt1 and pt2

        Vec2 proj = pt1 + r * L;
        return (p - proj).mag ();
      }
    }
  }



    if (deltaX () == 0.0) {
      return std::fabs (p.getX () - pt1.getX ());

    } else if (deltaY () == 0.0) {
      return std::fabs (p.getY () - pt1.getY ());

    } else {

      double a = 0.0;
      double b = 0.0;
      double c = 0.0;

      getABC (a, b, c);

      double denom = std::sqrt (a*a + b*b); 
      assert (denom != 0.0);

      return std::fabs (a * p.getX () + b * p.getY ()  + c) / denom;
    }

  }

}; // end LineSegment




#endif // _VEC2_H_
