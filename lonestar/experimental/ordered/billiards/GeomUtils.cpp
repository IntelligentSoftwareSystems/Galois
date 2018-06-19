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

#include "GeomUtils.h"
#include "FPutils.h"

Vec2 operator+(const Vec2& v1, const Vec2& v2) {

  Vec2 sum(v1);
  sum += v2;
  return sum;
}

Vec2 operator-(const Vec2& v1, const Vec2& v2) {

  Vec2 diff(v1);
  diff -= v2;
  return diff;
}

Vec2 operator/(const Vec2& v1, const FP& s) {

  Vec2 sv(v1);
  sv /= s;
  return sv;
}

Vec2 operator*(const Vec2& v1, const FP& s) {

  Vec2 sv(v1);
  sv *= s;
  return sv;
}

Vec2 operator*(const FP& s, const Vec2& v1) { return v1 * s; }

std::ostream& operator<<(std::ostream& out, const Vec2& v) {
  return (out << v.str());
}
