/*
 * Tuple.h
 *
 *  Created on: Jan 25, 2011
 *      Author: xinsui
 */

#ifndef TUPLE_H_
#define TUPLE_H_

#include <iostream>
class Tuple {
  double _t[2];
  long _id;
public:
  Tuple(double x, double y, long id) { _t[0] = x; _t[1] = y; _id = id; }
  Tuple() { }
  ~Tuple() { }

  inline double x() const { return _t[0]; }
  inline double y() const { return _t[1]; }
  inline long id() const { return _id; }
  bool operator==(const Tuple& rhs) const {
    for (int x = 0; x < 2; ++x) {
      if (_t[x] != rhs._t[x]) return false;
    }
    return true;
  }

  bool operator!=(const Tuple& rhs) const {
    return !(*this == rhs);
  }

  void print(std::ostream& os) const {
    os << "(" << _t[0] << ", " << _t[1] << ")";
  }
};

static inline std::ostream& operator<<(std::ostream& os, const Tuple& rhs) {
  rhs.print(os);
  return os;
}
#endif /* TUPLE_H_ */
