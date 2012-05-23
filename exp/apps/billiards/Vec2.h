/** 2D Point  -*- C++ -*-
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
 * 2D Point .
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



#endif // _VEC2_H_
