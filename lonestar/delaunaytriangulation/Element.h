#ifndef ELEMENT_H
#define ELEMENT_H

#include "Tuple.h"

#include <ostream>
#include <stdlib.h>

class Point;

class Element {
  Point* points[3];
  
public:
  Element(const Element& e) {
    points[0] = e.points[0];
    points[1] = e.points[1];
    points[2] = e.points[2];
  }

  Element(Point* a, Point* b, Point* c) {
    points[0] = a;
    points[1] = b;
    points[2] = c;
  }

  Element(Point* a, Point* b) {
    points[0] = a;
    points[1] = b;
    points[2] = NULL;
  }
  
  Point* getPoint(int i) { return points[i]; }
  const Point* getPoint(int i) const { return points[i]; }

  bool boundary() const { return points[2] == NULL; }
  int dim() const { return boundary() ? 2 : 3; }

  bool clockwise() const;
  
  //! determine if a tuple is inside the triangle
  bool inTriangle(const Tuple& p) const;

  //! determine if the circumcircle of the triangle contains the tuple
  bool inCircle(const Tuple& p) const;

  std::ostream& print(std::ostream& out) const;
};

std::ostream& operator<<(std::ostream& out, const Element& e);

#endif
