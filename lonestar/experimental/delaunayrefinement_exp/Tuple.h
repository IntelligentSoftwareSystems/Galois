/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
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

#ifndef TUPLE_H
#define TUPLE_H

#include <ostream>
#include <cmath>
#include "galois/runtime/Context.h"
#include "galois/runtime/Serialize.h"

class Tuple : public galois::runtime::Lockable {
  double _t[2];

public:
  Tuple(double a, double b) noexcept {
    _t[0] = a;
    _t[1] = b;
  }

  Tuple() = default;

  // serialization functions
  typedef int tt_has_serialize;
  void serialize(galois::runtime::SerializeBuffer& s) const {
    gSerialize(s, _t[0]);
    gSerialize(s, _t[1]);
  }
  void deserialize(galois::runtime::DeSerializeBuffer& s) {
    gDeserialize(s, _t[0]);
    gDeserialize(s, _t[1]);
  }

  bool operator==(const Tuple& rhs) const {
    for (int x = 0; x < 2; ++x) {
      if (_t[x] != rhs._t[x])
        return false;
    }
    return true;
  }

  bool operator!=(const Tuple& rhs) const { return !(*this == rhs); }

  bool operator<(const Tuple& rhs) const {
    for (int i = 0; i < 2; ++i) {
      if (_t[i] < rhs._t[i])
        return true;
      else if (_t[i] > rhs._t[i])
        return false;
    }
    return false;
  }

  bool operator>(const Tuple& rhs) const {
    for (int i = 0; i < 2; ++i) {
      if (_t[i] > rhs._t[i])
        return true;
      else if (_t[i] < rhs._t[i])
        return false;
    }
    return false;
  }

  Tuple operator+(const Tuple& rhs) const {
    return Tuple(_t[0] + rhs._t[0], _t[1] + rhs._t[1]);
  }

  Tuple operator-(const Tuple& rhs) const {
    return Tuple(_t[0] - rhs._t[0], _t[1] - rhs._t[1]);
  }

  Tuple operator*(double d) const { // scalar product
    return Tuple(_t[0] * d, _t[1] * d);
  }

  double operator*(const Tuple& rhs) const { // dot product
    return _t[0] * rhs._t[0] + _t[1] * rhs._t[1];
  }

  double operator[](int i) const { return _t[i]; };

  int cmp(const Tuple& x) const {
    if (*this == x)
      return 0;
    if (*this > x)
      return 1;
    return -1;
  }

  double distance_squared(
      const Tuple& p) const { // squared distance between current tuple and x
    double sum = 0.0;
    for (int i = 0; i < 2; ++i) {
      double d = _t[i] - p._t[i];
      sum += d * d;
    }
    return sum;
  }

  double distance(const Tuple& p) const { // distance between current tuple and
                                          // x
    return sqrt(distance_squared(p));
  }

  double angle(const Tuple& a,
               const Tuple& b) const { // angle formed by a, current tuple, b
    Tuple vb  = a - *this;
    Tuple vc  = b - *this;
    double dp = vb * vc;
    double c  = dp / sqrt(distance_squared(a) * distance_squared(b));
    return (180 / M_PI) * acos(c);
  }

  void angleCheck(const Tuple& a, const Tuple& b, bool& ob, bool& sm,
                  double M) const { // angle formed by a, current tuple, b
    Tuple vb  = a - *this;
    Tuple vc  = b - *this;
    double dp = vb * vc;

    if (dp < 0) {
      ob = true;
      return;
    }

    double c = dp / sqrt(distance_squared(b) * distance_squared(a));
    if (c > cos(M * M_PI / 180)) {
      sm = true;
      return;
    }
    return;
  }

  bool angleGTCheck(const Tuple& a, const Tuple& b,
                    double M) const { // angle formed by a, current tuple, b
    Tuple vb  = a - *this;
    Tuple vc  = b - *this;
    double dp = vb * vc;

    if (dp < 0)
      return false;

    double c = dp / sqrt(distance_squared(b) * distance_squared(a));
    return c > cos(M * M_PI / 180);
  }

  bool
  angleOBCheck(const Tuple& a,
               const Tuple& b) const { // angle formed by a, current tuple, b
    Tuple vb  = a - *this;
    Tuple vc  = b - *this;
    double dp = vb * vc;

    return dp < 0;
  }

  void print(std::ostream& os) const {
    os << "(" << _t[0] << ", " << _t[1] << ")";
  }

  static int cmp(Tuple a, Tuple b) { return a.cmp(b); }
  static double distance(Tuple a, Tuple b) { return a.distance(b); }
  static double angle(const Tuple& a, const Tuple& b, const Tuple& c) {
    return b.angle(a, c);
  }
  static void angleCheck(const Tuple& a, const Tuple& b, const Tuple& c,
                         bool& ob, bool& sm, double M) {
    b.angleCheck(a, c, ob, sm, M);
  }
  static bool angleGTCheck(const Tuple& a, const Tuple& b, const Tuple& c,
                           double M) {
    return b.angleGTCheck(a, c, M);
  }
  static bool angleOBCheck(const Tuple& a, const Tuple& b, const Tuple& c) {
    return b.angleOBCheck(a, c);
  }
};

static inline std::ostream& operator<<(std::ostream& os, const Tuple& rhs) {
  rhs.print(os);
  return os;
}

static inline Tuple operator*(double d, Tuple rhs) { return rhs * d; }

#endif
