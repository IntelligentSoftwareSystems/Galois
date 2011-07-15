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

/**
 * GReducible stores per thread values
 * of a variable of type T
 *
 * At the end of a for_each section, the final value is obtained by performing
 * a reduction on per thread values using the
 * provided binary functor BinFunc
 * 
 */

template <typename T, typename BinFunc>
class GReducible: public GaloisRuntime::PerCPU_merge<T> {
  typedef GaloisRuntime::PerCPU_merge<T> SuperType;

  static BinFunc _func;

  static BinFunc getFunc () {
    return _func;
  }

  static void reduce (T& lhs, T& rhs) {
    lhs = getFunc() (lhs, rhs);
  }

public:
  /**
   * @param val initial per thread value
   * @param func the binary functor acting as the reduction operator
   */
  explicit GReducible (const T& val = T(), BinFunc func = BinFunc()) 
    : GaloisRuntime::PerCPU_merge<T> (reduce, val) {
    _func = func;
  }

  /**
   * updates the thread local value
   * by applying the reduction operator to 
   * current and newly provided value
   *
   * @param _newVal
   */
  const T& update (const T& _newVal) {
    reduce (SuperType::get (), _newVal);
    return SuperType::get ();
  }
};


typedef GReducible<int, std::plus<int> > PerCPUcounter;

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
