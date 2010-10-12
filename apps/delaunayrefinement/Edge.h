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
 
 File: Edge.h
 
 Modified: February 14th, 2008 by Milind Kulkarni (initial version)
 Modified: October  5th, 2010 by Andrew Lenharth

 */ 
#ifndef _EDGE_H
#define _EDGE_H

#include "Tuple.h"

class Element;

class Edge {
  
  Tuple p[2];
  
 public:
  Edge() {}
  Edge(const Tuple& a, const Tuple& b)
  {
    if (a < b) {
      p[0] = a;
      p[1] = b;
    } else {
      p[0] = b;
      p[1] = a;
    }
  }
  Edge(const Edge &rhs) {
    p[0] = rhs.p[0];
    p[1] = rhs.p[1];
  }
  
  bool operator==(const Edge& rhs) const {
    return p[0] == rhs.p[0] && p[1] == rhs.p[1];
  }    
  bool operator!=(const Edge& rhs) const {
    return !(*this == rhs);
  }    
  bool operator<(const Edge& rhs) const {
    return ((p[0] < rhs.p[0]) || ((p[0] == rhs.p[0]) && (p[1] < rhs.p[1])));
  }    
  
  bool operator>(const Edge& rhs) const {
    return ((p[0] > rhs.p[0]) || ((p[0] == rhs.p[0]) && (p[1] > rhs.p[1])));
  }    

  Tuple getPoint(int i) const {
    return p[i];
  }
};
#endif
