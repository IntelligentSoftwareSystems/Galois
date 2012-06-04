/** Accumulator type -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_ACCUMULATOR_H
#define GALOIS_ACCUMULATOR_H

#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/LoopHooks.h"

namespace Galois {

// TODO(ddn): Rename to Galois/Reducers.h

/**
 * GReducible stores per thread values of a variable of type T
 *
 * At the end of a for_each section, the final value is obtained by performing
 * a reduction on per thread values using the provided binary functor BinFunc
 */
template <typename T, typename BinFunc>
class GReducible : public GaloisRuntime::AtLoopExit {
  BinFunc _func;
  GaloisRuntime::PerCPU<T> _data;

  void reduce() {
    T& d0 = _data.get(0);
    for (unsigned int i = 1; i < _data.size(); ++i)
      _func(d0, _data.get(i));
  }

  virtual void LoopExit() {
    reduce();
  }

public:
  typedef GReducible<T,BinFunc> SelfTy;

  /**
   * @param val initial per thread value
   */
  explicit GReducible(const T& val)
    : _data(val) { }
  /**
   * @param F the binary functor acting as the reduction operator
   */
  explicit GReducible(BinFunc F)
    : _func(F) { }
  /**
   * @param val initial per thread value
   * @param F the binary functor acting as the reduction operator
   */
  GReducible(const T& val, BinFunc F)
    : _func(F), _data(val) { }

  GReducible() {}

  /**
   * updates the thread local value
   * by applying the reduction operator to 
   * current and newly provided value
   *
   * @param _newVal
   */
  const T& update(const T& _newVal) {
    T& lhs = _data.get();
    _func(lhs, _newVal);
    return lhs;
  }

  /**
   * returns the thread local value if in a parallel loop or
   * the final reduction if in serial mode
   */
  T& get() {
    return _data.get();
  }

  /**
   * reset thread local value to the arg provided
   *
   * @param d
   */
  void reset(const T& d) {
    _data.reset(d);
  }
};


//Derived types

template<typename BinFunc>
struct ReduceAssignWrap {
  BinFunc F;
  ReduceAssignWrap(BinFunc f = BinFunc()) :F(f) {}
  template<typename T>
  void operator()(T& lhs, const T& rhs) {
    lhs = F(lhs, rhs);
  }
};

template<typename T>
class GAccumulator : public GReducible<T, ReduceAssignWrap<std::plus<T> > > {
  typedef GReducible<T, ReduceAssignWrap<std::plus<T> > > SuperType;
  using GReducible<T, ReduceAssignWrap<std::plus<T> > >::update;
public:
  explicit GAccumulator(const T& val = T()): SuperType(val) {}

  GAccumulator& operator+=(const T& rhs) {
    update(rhs);
    return *this;
  }

  GAccumulator& operator-=(const T& rhs) {
    update(-rhs);
    return *this;
  }
};

namespace HIDDEN {
template<typename T>
struct gmax {
  void operator()(T& lhs, const T& rhs) const {
    lhs = std::max(lhs,rhs);
  }
};
}

template<typename T>
class GReduceMax : public GReducible<T, HIDDEN::gmax<T> > {
  typedef GReducible<T, HIDDEN::gmax<T> > Super;
public:
  explicit GReduceMax(const T& val = T()): Super(val) {}
};

template<typename T>
class GReduceAverage {
  typedef std::pair<T, unsigned> TP;
  struct AVG {
    void operator() (TP& lhs, const TP& rhs) const {
      lhs.first += rhs.first;
      lhs.second += rhs.second;
    }
  };
  GReducible<std::pair<T, unsigned>, AVG> data;

public:
  void update(const T& _newVal) {
    data.update(std::make_pair(_newVal, 1));
  }

  /**
   * returns the thread local value if in a parallel loop or
   * the final reduction if in serial mode
   */
  const T get() {
    const TP& d = data.get();
    return d.first / d.second;
  }

  void reset(const T& d) {
    data.reset(std::make_pair(d, 0));
  }

  GReduceAverage& insert(const T& rhs) {
    TP& d = data.get();
    d.first += rhs;
    d.second++;
    return *this;
  }
};



/**
 * An alternate implementation of GReducible,
 * where 
 * - the final reduction does not automatically 
 * happen and does not over-write the value for thread 0
 * - copy construction is allowed
 * - simple std binary functors allowed
 */
template <typename T, typename BinFunc>
class GSimpleReducible: protected GaloisRuntime::PerCPU<T> {
  typedef GaloisRuntime::PerCPU<T> SuperType;
  BinFunc func;

public:
  explicit GSimpleReducible (const T& val = T(), BinFunc func=BinFunc())
    : GaloisRuntime::PerCPU<T> (val), func(func)  {}


  T reduce () const {
    T val (SuperType::get (0));

    for (unsigned i = 1; i < SuperType::size (); ++i) {
      val = func (val, SuperType::get (i));
    }

    return val;
  }

  const T& update (const T& _newVal) {
    T& oldVal = SuperType::get ();
    oldVal = func (oldVal, _newVal);
    return oldVal;
  }

  T& get () {
    return SuperType::get ();
  }

  const T& get () const {
    return SuperType::get ();
  }

  void reset (const T& val) {
    SuperType::reset (val);
  }

};




}

#endif
