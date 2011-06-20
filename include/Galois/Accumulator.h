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

#include "Runtime/PerCPU.h"

namespace Galois {

template<typename T>
class accumulator {
  GaloisRuntime::PerCPU_merge<T> data;

  static void acc(T& lhs, T& rhs) {
    lhs += rhs;
    rhs = 0;
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

}

#endif
