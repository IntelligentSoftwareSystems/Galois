/** Delaunay refinement -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * @author Milind Kulkarni <milind@purdue.edu>>
 */
#ifndef _ELEMENT_H
#define _ELEMENT_H

#include <cassert>
#include <stdlib.h>

#include "Edge.h"
#include "Galois/Runtime/Serialize.h"

#define MINANGLE 30.0

class Element : public galois::runtime::Lockable {
  Tuple coords[3]; // The three endpoints of the triangle
  // if the triangle has an obtuse angle
  // obtuse - 1 is which one
  signed char obtuse;
  bool bDim; // true == 3, false == 2
  int id;

public:

//NOTE!!! serialize and deserialize the data
  // serialization functions
  typedef int tt_has_serialize;
  void serialize(galois::runtime::SerializeBuffer& s) const {
    gSerialize(s,coords[0]);
    gSerialize(s,coords[1]);
    gSerialize(s,coords[2]);
    gSerialize(s,obtuse);
    gSerialize(s,bDim);
    gSerialize(s,id);
  }
  void deserialize(galois::runtime::DeSerializeBuffer& s) {
    gDeserialize(s,coords[0]);
    gDeserialize(s,coords[1]);
    gDeserialize(s,coords[2]);
    gDeserialize(s,obtuse);
    gDeserialize(s,bDim);
    gDeserialize(s,id);
  }

  // required by the new in DataLandingPad in Directory.h
  Element() = default; //noexcept :obtuse(0), bDim(true), id(0) {}

  //! Constructor for Triangles
  Element(const Tuple& a, const Tuple& b, const Tuple& c, int _id = 0) noexcept
   :obtuse(0), bDim(true), id(_id) 
  { 
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
    for (int i = 0; i < 3; i++)
      if (angleOBCheck(i))
	obtuse = i + 1;
    //computeCenter();
  }

  //! Constructor for segments
  Element(const Tuple& a, const Tuple& b, int _id = 0) noexcept
  : obtuse(0), bDim(false), id(_id) {
    coords[0] = a;
    coords[1] = b;
    if (b < a) {
      coords[0] = b;
      coords[1] = a;
    }
    //computeCenter();
  }

  Tuple getCenter() const {
    if (dim() == 2) {
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

  bool operator==(const Element& rhs) const {
    return id == rhs.id;
  }

  bool operator<(const Element& rhs) const {
    //apparently a triangle is less than a line
    if (dim() < rhs.dim()) return false;
    if (dim() > rhs.dim()) return true;
    for (int i = 0; i < dim(); i++) {
      if (coords[i] < rhs.coords[i]) return true;
      else if (coords[i] > rhs.coords[i]) return false;
    }
    return false;
  }

  /// @return if the current triangle has a common edge with e
  bool isRelated(const Element& rhs) const {
    int num_eq = 0;
    for(int i = 0; i < dim(); ++i)
      for(int j = 0; j < rhs.dim(); ++j)
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
    int j = (i + 1) % dim();
    int k = (i + 2) % dim(); 
    Tuple::angleCheck(coords[j], coords[i], coords[k], ob, sm, M);
  }

  bool angleGTCheck(int i, double M) const {
    int j = (i + 1) % dim();
    int k = (i + 2) % dim(); 
    return Tuple::angleGTCheck(coords[j], coords[i], coords[k], M);
  }
  
  bool angleOBCheck(int i) const {
    int j = (i + 1) % dim();
    int k = (i + 2) % dim(); 
    return Tuple::angleOBCheck(coords[j], coords[i], coords[k]);
  }

  //Virtualize the Edges array
  //Used only by Mesh now
  Edge getEdge(int i) const {
    if (i == 0)
      return Edge(coords[0], coords[1]);
    if (!bDim) {
      if (i == 1)
	return Edge(coords[1], coords[0]);
    } else {
      if (i == 1)
	return Edge(coords[1], coords[2]);
      else if (i == 2)
	return Edge(coords[2], coords[0]);
    }
    GALOIS_DIE("unknown edge");
    return Edge(coords[0], coords[0]);
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
    }
    GALOIS_DIE("no obtuse edge");
    return getEdge(0);
  }

  //! Should the node be processed?
  bool isBad() const {
    if (!bDim)
      return false;
    for (int i = 0; i < 3; i++)
      if (angleGTCheck(i, MINANGLE))
	return true;
    return false;
  }

  const Tuple& getPoint(int i) const { return coords[i]; }

  const Tuple& getObtuse() const { return coords[obtuse-1]; }

  int dim() const { return bDim ? 3 : 2; }

  int numEdges() const { return dim() + dim() - 3; }

  bool isObtuse() const { return obtuse != 0; }

  int getId() const { return id; }

  /**
   * Scans all the edges of the two elements and if it finds one that is
   * equal, then sets this as the Edge of the EdgeRelation
   */
  Edge getRelatedEdge(const Element& e) const {
    int at = 0;
    Tuple d[2];
    for(int i = 0; i < dim(); ++i)
      for(int j = 0; j < e.dim(); ++j)
        if (coords[i] == e.coords[j])
	  d[at++] = coords[i];
    assert(at == 2);
    return Edge(d[0], d[1]);
  }

  std::ostream& print(std::ostream& s) const {
    s << '[';
    for (int i = 0; i < dim(); ++i)
      s << coords[i] << (i < (dim() - 1) ? ", " : "");
    s << ']';
    return s;
  }
};

static std::ostream& operator<<(std::ostream& s, const Element& E) {
  return E.print(s);
}

#endif
