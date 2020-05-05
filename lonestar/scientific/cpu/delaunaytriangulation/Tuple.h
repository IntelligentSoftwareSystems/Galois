/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef TUPLE_H
#define TUPLE_H

#include <ostream>
#include <cmath>

typedef double TupleDataTy;

class Tuple {
  TupleDataTy data[2];

public:
  Tuple() {
    data[0] = 0;
    data[1] = 0;
  }
  Tuple(TupleDataTy xy) {
    data[0] = xy;
    data[1] = xy;
  }
  Tuple(TupleDataTy x, TupleDataTy y) {
    data[0] = x;
    data[1] = y;
  }
  int dim() const { return 2; }
  TupleDataTy x() const { return data[0]; }
  TupleDataTy y() const { return data[1]; }

  TupleDataTy& x() { return data[0]; }
  TupleDataTy& y() { return data[1]; }

  bool operator==(const Tuple& rhs) const {
    for (int i = 0; i < 2; ++i) {
      if (data[i] != rhs.data[i])
        return false;
    }
    return true;
  }

  bool operator!=(const Tuple& rhs) const { return !(*this == rhs); }

  TupleDataTy operator[](int index) const { return data[index]; }

  TupleDataTy& operator[](int index) { return data[index]; }

  Tuple operator+(const Tuple& rhs) const {
    return Tuple(data[0] + rhs.data[0], data[1] + rhs.data[1]);
  }

  Tuple operator-(const Tuple& rhs) const {
    return Tuple(data[0] - rhs.data[0], data[1] - rhs.data[1]);
  }

  //! scalar product
  Tuple operator*(TupleDataTy d) const {
    return Tuple(data[0] * d, data[1] * d);
  }

  //! dot product
  TupleDataTy dot(const Tuple& rhs) const {
    return data[0] * rhs.data[0] + data[1] * rhs.data[1];
  }

  TupleDataTy cross(const Tuple& rhs) const {
    return data[0] * rhs.data[1] - data[1] * rhs.data[0];
  }

  void print(std::ostream& os) const {
    os << "(" << data[0] << ", " << data[1] << ")";
  }
};

static inline std::ostream& operator<<(std::ostream& os, const Tuple& rhs) {
  rhs.print(os);
  return os;
}

class Tuple3 {
  TupleDataTy data[3];

public:
  Tuple3() {
    data[0] = 0;
    data[1] = 0;
    data[2] = 0;
  }
  Tuple3(TupleDataTy xyz) {
    data[0] = xyz;
    data[1] = xyz;
    data[1] = xyz;
  }
  Tuple3(TupleDataTy x, TupleDataTy y, TupleDataTy z) {
    data[0] = x;
    data[1] = y;
    data[2] = z;
  }
  int dim() const { return 3; }
  TupleDataTy x() const { return data[0]; }
  TupleDataTy y() const { return data[1]; }
  TupleDataTy z() const { return data[2]; }

  TupleDataTy& x() { return data[0]; }
  TupleDataTy& y() { return data[1]; }
  TupleDataTy& z() { return data[2]; }

  bool operator==(const Tuple3& rhs) const {
    for (int i = 0; i < 3; ++i) {
      if (data[i] != rhs.data[i])
        return false;
    }
    return true;
  }

  bool operator!=(const Tuple3& rhs) const { return !(*this == rhs); }

  TupleDataTy operator[](int index) const { return data[index]; }

  TupleDataTy& operator[](int index) { return data[index]; }

  Tuple3 operator+(const Tuple3& rhs) const {
    return Tuple3(data[0] + rhs.data[0], data[1] + rhs.data[1],
                  data[2] + rhs.data[2]);
  }

  Tuple3 operator-(const Tuple3& rhs) const {
    return Tuple3(data[0] - rhs.data[0], data[1] - rhs.data[1],
                  data[2] + rhs.data[2]);
  }

  //! scalar product
  Tuple3 operator*(TupleDataTy d) const {
    return Tuple3(data[0] * d, data[1] * d, data[2] * d);
  }

  //! dot product
  TupleDataTy dot(const Tuple3& rhs) const {
    return data[0] * rhs.data[0] + data[1] * rhs.data[1] +
           data[2] * rhs.data[2];
  }

  Tuple3 cross(const Tuple3& rhs) const {
    return Tuple3(data[1] * rhs.data[2] - data[2] * rhs.data[1],
                  data[2] * rhs.data[0] - data[0] * rhs.data[2],
                  data[0] * rhs.data[1] - data[1] * rhs.data[0]);
  }

  void print(std::ostream& os) const {
    os << "(" << data[0] << ", " << data[1] << ", " << data[2] << ")";
  }
};

static inline std::ostream& operator<<(std::ostream& os, const Tuple3& rhs) {
  rhs.print(os);
  return os;
}

#endif
