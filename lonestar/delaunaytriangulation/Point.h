#ifndef POINT_H
#define POINT_H

#include "Tuple.h"
#include "Graph.h"

#include "galois/CheckedObject.h"

#include <ostream>
#include <algorithm>

class Point: public galois::GChecked<void> {
  Tuple m_t;
  GNode m_n;
  long m_id;
  
public:
  Point(double x, double y, long id): m_t(x,y), m_n(NULL), m_id(id) {}

  const Tuple& t() const { return m_t; }
  long id() const { return m_id; }

  Tuple& t() { return m_t; }
  long& id() { return m_id; }

  void addElement(const GNode& n) {
    m_n = n;
  }

  void removeElement(const GNode& n) {
    if (m_n == n)
      m_n = NULL;
  }

  bool inMesh() const {
    return m_n != NULL;
  }

  GNode someElement() const {
    return m_n;
  }

  void print(std::ostream& os) const {
    os << "(id: " << m_id << " t: ";
    m_t.print(os);
    if (m_n != NULL)
      os << " SOME)";
    else
      os << " NULL)";
  }
};

static inline std::ostream& operator<<(std::ostream& os, const Point& rhs) {
  rhs.print(os);
  return os;
}

#endif
