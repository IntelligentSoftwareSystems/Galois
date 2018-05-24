#ifndef EDGE_H
#define EDGE_H

#include "Tuple.h"

class Element;

class Edge {
  Tuple p[2];
  
public:
  Edge() {}
  Edge(const Tuple& a, const Tuple& b) {
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
    return (p[0] < rhs.p[0]) || ((p[0] == rhs.p[0]) && (p[1] < rhs.p[1]));
  }    
  
  bool operator>(const Edge& rhs) const {
    return (p[0] > rhs.p[0]) || ((p[0] == rhs.p[0]) && (p[1] > rhs.p[1]));
  }    

  Tuple getPoint(int i) const {
    return p[i];
  }
};
#endif
