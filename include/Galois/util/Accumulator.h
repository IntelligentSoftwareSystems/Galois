// Accumulator type -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

#ifndef __GALOIS_ACCUMULATOR_H
#define __GALOIS_ACCUMULATOR_H

#include "Galois/Runtime/PerCPU.h"

namespace Galois {

template<typename T>
class accumulator {
  GaloisRuntime::PerCPU_merge<T> data;

  static void acc(T& lhs, T& rhs) {
    lhs += rhs;
  }

public:
  accumulator() :data(acc) {}

  accumulator& operator+=(const T& rhs) {
    data.get() += rhs;
    return *this;
  }

  accumulator& operator-=(const T& rhs) {
    data.get() -= rhs;
    return *this;
  }
 
  const T& get() const {
    return data.get();
  }

  void reset(const T& d) {
    data.reset(d);
  }

};

template<typename T>
class reduce_max {
  GaloisRuntime::PerCPU_merge<T> data;

  static void acc(T& lhs, T& rhs) {
    lhs = std::max(lhs, rhs);
  }
public:
  reduce_max() :data(acc) {}

  reduce_max& insert(const T& rhs) {
    T& d = data.get();
    if (d < rhs)
      d = rhs;
    return *this;
  }

  const T& get() const {
    return data.get();
  }

  void reset(const T& d) {
    data.reset(d);
  }


};

template<typename T>
class reduce_average {
  typedef std::pair<T, unsigned> TP;
  GaloisRuntime::PerCPU_merge<TP> data;

  static void acc(TP& lhs, TP& rhs) {
    lhs.first += rhs.first;
    lhs.second += rhs.second;
  }
public:
  reduce_average() :data(acc) {}

  reduce_average& insert(const T& rhs) {
    TP& d = data.get();
    d.first += rhs;
    d.second++;
    return *this;
  }

  T get() const {
    const TP& d = data.get();
    return d.first / d.second;
  }

  void reset(const T& d) {
    data.reset(std::make_pair(d,0) );
  }
};

}

#endif
