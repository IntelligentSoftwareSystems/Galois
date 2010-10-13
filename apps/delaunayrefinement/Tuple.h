/* 
 
 Lonestar DelaunayRefinement: Refinement of an initial, unrefined Delaunay
 mesh to eliminate triangles with angles < 30 degrees, using a
 variation of Chew's algorithm.
 
 Authors: Milind Kulkarni 
 
 Copyright (C) 2007, 2008 The University of Texas at Austin
 
 Licensed under the Eclipse Public License, Version 1.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.eclipse.org/legal/epl-v10.html
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 
 File: Tuple.h
 
 Modified: December 2nd, 2007 by Milind Kulkarni (initial version)
 Modified: October  5th, 2010 by Andrew Lenharth
 
 */ 

#ifndef _TUPLE_H_
#define _TUPLE_H_

#include <cstdio>
#include <cmath>
#include <iostream>

class Tuple {
  double _t[3];

public:
  
  Tuple(double a, double b, double c) {
    _t[0] = a;
    _t[1] = b;
    _t[2] = c;
  }

  Tuple() {};
  ~Tuple() {};
  
  bool operator==(const Tuple& rhs) const {
    return (_t[0] == rhs._t[0]) && (_t[1] == rhs._t[1]) && (_t[2] == rhs._t[2]);
  }

  bool operator!=(const Tuple& rhs) const {
    return !(*this == rhs);
  }

  bool operator<(const Tuple& rhs) const {
    for (int i = 0; i < 3; ++i)
      if (_t[i] < rhs._t[i]) return true;
      else if (_t[i] > rhs._t[i]) return false;
    return false;
  }

  bool operator>(const Tuple& rhs) const {
    for (int i = 0; i < 3; ++i)
      if (_t[i] > rhs._t[i]) return true;
      else if (_t[i] < rhs._t[i]) return false;
    return false;
  }
  
  Tuple operator+(const Tuple& rhs) const {
    return Tuple(_t[0]+rhs._t[0], _t[1]+rhs._t[1], _t[2]+rhs._t[2]);
  }

  Tuple operator-(const Tuple& rhs) const {
    return Tuple(_t[0]-rhs._t[0], _t[1]-rhs._t[1], _t[2]-rhs._t[2]);
  }

  Tuple operator*(double d) const { //scalar product
    return Tuple(_t[0]*d, _t[1]*d, _t[2]*d);
  }

  double operator*(const Tuple& rhs) const { //dot product
    return _t[0]*rhs._t[0] + _t[1]*rhs._t[1] + _t[2]*rhs._t[2];
  }

  double operator[](int i) const {
    return _t[i];
  };
  
  int cmp(const Tuple& x) const {
    if (*this == x)
      return 0;
    if (*this > x)
      return 1;
    return -1;
  }

  double distance_squared(const Tuple& p) const { //squared distance between current tuple and x
    double x = _t[0] - p._t[0];
    double y = _t[1] - p._t[1];
    double z = _t[2] - p._t[2];
    return x*x + y*y + z*z;
  }
  
  double distance(const Tuple& p) const { //distance between current tuple and x
    return sqrt(distance_squared(p));
  }
  
  double angle(const Tuple& a, const Tuple& b) const { //angle formed by a, current tuple, b
    Tuple vb = a - *this;
    Tuple vc = b - *this;
    double dp = vb*vc;
    double c = dp / sqrt(distance_squared(a) * distance_squared(b));
    return (180/M_PI) * acos(c);
  }

  void print(std::ostream& os) const {
    char *buf = new char[256];
    sprintf(buf, "(%.4f, %.4f, %.4f)", _t[0], _t[1], _t[2]);
    os << buf;
  }
  
  static int cmp(Tuple a, Tuple b) {return a.cmp(b);};
  static double distance(Tuple a, Tuple b) {return a.distance(b);};
  static double angle(Tuple a, Tuple b, Tuple c) {return b.angle(a, c);};
  
};

static inline std::ostream& operator<<(std::ostream& os, const Tuple& rhs) {
  rhs.print(os);
  return os;
}

static inline Tuple operator*(double d, Tuple rhs) {
  return rhs * d;
}


#endif
