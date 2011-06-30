/** A tuple -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * @author Xin Sui <xinsui@cs.utexas.edu>
 */
#ifndef TUPLE_H_
#define TUPLE_H_

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
    for (int i = 0; i < 2; ++i) {
      if (_t[i] != rhs._t[i]) return false;
    }
    return true;
  }

  bool operator!=(const Tuple& rhs) const {
    return !(*this == rhs);
  }

  void print(std::ostream& os) const {
    os << "(" << _t[0] << ", " << _t[1] << " id: " << _id << ")";
  }
};

static inline std::ostream& operator<<(std::ostream& os, const Tuple& rhs) {
  rhs.print(os);
  return os;
}
#endif /* TUPLE_H_ */
