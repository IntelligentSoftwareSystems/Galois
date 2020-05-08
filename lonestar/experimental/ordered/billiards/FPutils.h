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

#ifndef _FP_UTILS_H_
#define _FP_UTILS_H_

#include <string>
#include <iostream>

#include <boost/noncopyable.hpp>
#include <boost/rational.hpp>

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cstdint>

#include "fixed_point.h"
#include "FPwrapper.h"

template <typename T>
struct FPutilsGeneric : private boost::noncopyable {

  static const T EPSILON;
  static const T ERROR_LIMIT;

  static T sqrt(const T& t) {
    double d = std::sqrt(double(t));
    T r      = T::sqrt(t);

    if (d != 0.0) {
      assert(std::fabs((r.dval() - d) / d) < 1e-5);
    }

    return r;
  }

  static bool almostEqual(const T& d1, const T& d2) {
    return (T::fabs(d1 - d2) < EPSILON);
  }

  template <typename V>
  static bool almostEqual(const V& v1, const V& v2) {
    return almostEqual(v1.getX(), v2.getX()) &&
           almostEqual(v1.getY(), v2.getY());
  }

  //! checks relative error
  static bool checkError(const T& original, const T& measured,
                         bool useAssert = true) {

    T err = T::fabs(measured - original);
    if (original != T(0.0)) {
      err = T::fabs((measured - original) / original);
    }

    bool withinLim = err < ERROR_LIMIT;

    if (!withinLim) {
      fprintf(stderr,
              "WARNING: FPutils::checkError, relative error=%10.20g above "
              "limit, ERROR_LIMIT=%10.20g\n",
              double(err), double(ERROR_LIMIT));
    }

    if (useAssert) {
      assert(withinLim && "Relative Error above limit?");
    }

    return withinLim;
  }

  template <typename V>
  static bool checkError(const V& original, const V& measured,
                         bool useAssert = true) {
    return checkError(original.getX(), measured.getX(), useAssert) &&
           checkError(original.getY(), measured.getY(), useAssert);
  }
};

template <typename T>
const T FPutilsGeneric<T>::EPSILON = double(1.0 / (1 << 30));

template <typename T>
const T FPutilsGeneric<T>::ERROR_LIMIT = double(1.0 / (1 << 18));

// using FP = fpml::fixed_point<int64_t, 31, 32>;
using FP      = DoubleWrapper;
using FPutils = FPutilsGeneric<FP>;

#endif // _FP_UTILS_H_
