// Element data in the graph -*- C++ -*-

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
 
   File: Element.h
 
   Created: February 5th, 2008 by Milind Kulkarni (initial version)
 
*/ 

#ifndef _ELEMENT_H
#define _ELEMENT_H

#include <cassert>
#include <stdlib.h>

#include "Edge.h"

#define MINANGLE 30.0

class Element {
  Tuple coords[3]; // The three endpoints of the triangle
  //the last index is the center

  // if the triangle has an obtuse angle
  // obtuse - 1 is which one
  signed char obtuse;
  bool bBad;
  bool bDim; // true == 3, false == 2

 public:
  
 explicit Element(const Tuple& a, const Tuple& b, const Tuple& c)
   :obtuse(0), bBad(0), bDim(true)
  { //constructor for Triangles
    coords[0] = a;
    coords[1] = b;
    coords[2] = c;
    if (b < a || c < a) {
      if (b < c) {
	coords[0] = b;
	coords[1] = c;
	coords[2] = a;
      } else {
	coords[0] = c;
	coords[1] = a;
	coords[2] = b;
      }
    }
    //    edges[0] = Edge(coords[0], coords[1]);
    //    edges[1] = Edge(coords[1], coords[2]);
    //    edges[2] = Edge(coords[2], coords[0]);
    for (int i = 0; i < 3; i++) {
      bool ob = false, sm = false;
      angleCheck(i, ob, sm, MINANGLE);
      if (ob) {
	obtuse = i + 1;
      } else if (sm) {
	bBad = true;
      }
    }
    //computeCenter();
  }
  
  explicit Element(const Tuple& a, const Tuple& b)
    :obtuse(0), bBad(0), bDim(false)
  { //constructor for segments
    coords[0] = a;
    coords[1] = b;
    if (b < a) {
      coords[0] = b;
      coords[1] = a;
    }
    //computeCenter();
  }

  // Tuple getCenter() const {
  //   return coords[3];
  // }

  Tuple getCenter() const {
    if (getDim() == 2) {
      return (coords[0] + coords[1]) * 0.5;
    } else {
      const Tuple& a = coords[0];
      const Tuple& b = coords[1];
      const Tuple& c = coords[2];
      Tuple x = b - a;
      Tuple y = c - a;
      double xlen = a.distance(b);
      double ylen = a.distance(c);
      double cosine = (x * y) / (xlen * ylen);
      double sine_sq = 1.0 - cosine * cosine;
      double plen = ylen / xlen;
      double s = plen * cosine;
      double t = plen * sine_sq;
      double wp = (plen - cosine) / (2 * t);
      double wb = 0.5 - (wp * s);
      Tuple tmpval = a * (1 - wb - wp);
      tmpval = tmpval + (b * wb);
      return tmpval + (c * wp);
    }
  }
  
  double get_radius_squared() const {
    return get_radius_squared(getCenter());
  }

  double get_radius_squared(const Tuple& center) const {
    return center.distance_squared(coords[0]);
  }

  bool operator< (const Element& rhs) const {
    //apparently a triangle is less than a line
    if (getDim() < rhs.getDim()) return false;
    if (getDim() > rhs.getDim()) return true;
    for (int i = 0; i < getDim(); i++) {
      if (coords[i] < rhs.coords[i]) return true;
      else if (coords[i] > rhs.coords[i]) return false;
    }
    return false;
  }

  /// @return if the current triangle has a common edge with e
  bool isRelated(const Element& rhs) const {
    int num_eq = 0;
    for(int i = 0; i < getDim(); ++i)
      for(int j = 0; j < rhs.getDim(); ++j)
	if (coords[i] == rhs.coords[j])
	  ++num_eq;
    return num_eq == 2;
  }

  bool inCircle(Tuple p) const {
    Tuple center = getCenter();
    double ds = center.distance_squared(p);
    return ds <= get_radius_squared(center);
  }

  void angleCheck(int i, bool& ob, bool& sm, double M) const {
    int j = (i + 1) % getDim();
    int k = (i + 2) % getDim(); 
    Tuple::angleCheck(coords[j], coords[i], coords[k], ob, sm, M);
  }

  //Virtualize the Edges array
  //Used only by Mesh now
  Edge getEdge(int i) const {
    if (!bDim) {
      if (i == 0)
	return Edge(coords[0], coords[1]);
      else if (i == 1)
	return Edge(coords[1], coords[0]);
    } else {
      if (i == 0)
	return Edge(coords[0], coords[1]);
      else if (i == 1)
	return Edge(coords[1], coords[2]);
      else if (i == 2)
	return Edge(coords[2], coords[0]);
    }
    assert(0 && "unknown edge");
    abort();
  }

  const Tuple& getPoint(int i) const {
    return coords[i];
  }

  const Tuple& getObtuse() const {
    return coords[obtuse-1];
  }

  Edge getOppositeObtuse() const {
    //The edge opposite the obtuse angle is the edge formed by
    //the other indexes
    switch (obtuse) {
    case 1:
      return getEdge(1);
    case 2:
      return getEdge(2);
    case 3:
      return getEdge(0);
    };
    assert(0 && "no obtuse edge");
    abort();
  }

  // should the node be processed?
  bool isBad() const {
    return bBad;
  }

  int getDim() const {
    return bDim ? 3 : 2;
  }

  int numEdges() const {
    return getDim() + getDim() - 3;
  }

  bool isObtuse() const {
    return obtuse != 0;
  }

  /**
   * Scans all the edges of the two elements and if it finds one that is
   * equal, then sets this as the Edge of the EdgeRelation
   */
  Edge getRelatedEdge(const Element& e) const {
    int at = 0;
    Tuple d[2];
    for(int i = 0; i < getDim(); ++i)
      for(int j = 0; j < e.getDim(); ++j)
	if (coords[i] == e.coords[j])
	  d[at++] = coords[i];
    assert(at == 2);
    return Edge(d[0], d[1]);
  }

  std::ostream& print(std::ostream& s) const {
    s << '[';
    for (int i = 0; i < getDim(); ++i)
      s << coords[i] << (i < (getDim() - 1) ? ", " : "");
    s << ']';
    return s;
  }

};

static std::ostream& operator<<(std::ostream& s, const Element& E) {
  return E.print(s);
}

#endif
