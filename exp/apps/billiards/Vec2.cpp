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


#include "Vec2.h"

Vec2 operator + (const Vec2& v1, const Vec2& v2) {

  Vec2 sum (v1);
  sum += v2;
  return sum;
}

Vec2 operator - (const Vec2& v1, const Vec2& v2) {

  Vec2 diff (v1);
  diff -= v2;
  return diff;
}

Vec2 operator / (const Vec2& v1, const double s) {

  Vec2 sv (v1);
  sv /= s;
  return sv;
}

Vec2 operator * (const Vec2& v1, const double s) {

  Vec2 sv (v1);
  sv *= s;
  return sv;
}

Vec2 operator * (const double s, const Vec2& v1) {
  return v1 * s;
}

std::ostream& operator << (std::ostream& out, const Vec2& v) {
  return (out << v.str ());
}


