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
  bool bObtuse; // if the triangle has an obtuse angle
  bool bBad;

  Tuple obtuse;
  Tuple coords[3]; // The three endpoints of the triangle
  Edge edges[3]; // The edges connecting it to neighboring triangles
  int dim; // (=3 is a triangle, =2 is a segment)

  Tuple center; // The coordinates of the center of the circumcircle the triangle
  double radius_squared;
  double minAngle;

 public:
	
  Element(Tuple a, Tuple b, Tuple c) { //constructor for Triangles
    dim = 3;
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
    edges[0] = Edge(coords[0], coords[1]);
    edges[1] = Edge(coords[1], coords[2]);
    edges[2] = Edge(coords[2], coords[0]);
    bool l_bObtuse = false;
    bool l_bBad = false;
    Tuple l_obtuse;
    minAngle = 180.0;
    for (int i = 0; i < 3; i++) {
      double angle = getAngle(i);
      if (angle > 90.1) {
        l_bObtuse = true;
        l_obtuse = coords[i];
      } else if (angle < MINANGLE) {
        l_bBad = true;
      }
      if (angle < minAngle) {
        minAngle = angle;
      }
    }
    bBad = l_bBad;
    bObtuse = l_bObtuse;
    obtuse = l_obtuse;
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
    center = tmpval + (c * wp);
    radius_squared = center.distance_squared(a);
  }

  Element(Tuple a, Tuple b) { //constructor for segments
    dim = 2;
    coords[0] = a;
    coords[1] = b;
    if (b < a) {
      coords[0] = b;
      coords[1] = a;
    }
    edges[0] = Edge(coords[0], coords[1]);
    edges[1] = Edge(coords[1], coords[0]);
    bBad = false;
    bObtuse = false;
    center = (a + b) * 0.5;
    radius_squared = center.distance_squared(a);
  }

  bool operator< (const Element& rhs) const {
    //apparently a triangle is less than a line
    if (dim < rhs.getDim()) return false;
    if (dim > rhs.getDim()) return true;
    for (int i = 0; i < dim; i++) {
      if (coords[i] < rhs.coords[i]) return true;
      else if (coords[i] > rhs.coords[i]) return false;
    }
    return false;
  }

  /// @return if the current triangle has a common edge with e 
  bool isRelated(const Element& rhs) const {
    for(int i = 0; i < dim; ++i)
      for(int j = 0; j < rhs.dim; ++j)
	if (edges[i] == rhs.edges[j]) 
	  return true;
    return false;  
  }

  const Tuple& getCenter() const {
    return center;
  }

  double getMinAngle() const {
    return minAngle;
  }

  bool inCircle(Tuple p) const {
    double ds = center.distance_squared(p);
    return ds <= radius_squared;
  }

  double getAngle(int i) const {
    int j = (i + 1) % dim;
    int k = (i + 2) % dim; 
    Tuple a = coords[i];
    Tuple b = coords[j];
    Tuple c = coords[k];
    return Tuple::angle(b, a, c);
  }

  const Edge& getEdge(int i) const {
    return edges[i];
  }

  const Tuple& getPoint(int i) const {
    return coords[i];
  }

  const Tuple& getObtuse() const {
    return obtuse;
  }

  // should the node be processed?
  bool isBad() const {
    return bBad;
  }

  int getDim() const {
    return dim;
  }

  int numEdges() const {
    return dim + dim - 3;
  }

  bool isObtuse() const {
    return bObtuse;
  }

  /**
   * Scans all the edges of the two elements and if it finds one that is
   * equal, then sets this as the Edge of the EdgeRelation
   */
  const Edge& getRelatedEdge(const Element& e) const {
    for(int i = 0; i < dim; ++i)
      for(int j = 0; j < e.dim; ++j)
        if (edges[i] == e.edges[j])
          return edges[i];
    assert(0);
    abort();
  }

  double getRadiusSquared() const {
    return radius_squared;
  }

  std::ostream& print(std::ostream& s) const {
    s << '[';
    for (int i = 0; i < dim; ++i)
      s << coords[i] << (i < (dim - 1) ? ", " : "");
    s << ']';
    return s;
  }

};

static std::ostream& operator<<(std::ostream& s, const Element& E) {
  return E.print(s);
}

#endif
