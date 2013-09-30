/** FP utils and constants  -*- C++ -*-
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
 * @section Description
 *
 * FP utils and constants.
 *
 * @author <ahassaan@ices.utexas.edu>
 */



#ifndef _FP_UTILS_H_
#define _FP_UTILS_H_

#include <string>
#include <iostream>

#include <boost/noncopyable.hpp>

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
// cstdint requires c++0x
#include <stdint.h>

#include "Vec2.h"

struct FPutils: private boost::noncopyable {

  static const double EPSILON;

  // We want to truncate double such that results of simple operations e.g.
  // addition/multiplication does not get affected by rounding.
  //
  // - First we truncat the inputs to an opteration. 
  // - In addition, the difference between exponents dictates how much the smaller 
  // quantity will be shifted to the right. This means that if largest possible difference between
  // exponents of two number is X then we should truncate the fraction with X zeros to the right
  //
  // - In multiplication, the number of non-zero digits in fractional part of the result is
  // the sum of the nuber of digits in fractional part of operands. If the machine format allows N
  // bits of fraction, the the operands should be truncated to N/2 bits of fraction so that the
  // result doesn't get rounded. Intel's 80-bit format allows 64 bits of fraction, which means
  // operands should be truncated to 32 bits of fraction
  //
  // - The result of division and square root can have infinite digits in fraction
  //
  // 32 bits give us a precision of 2^(-32) = 2.0e-10
private:
  static const unsigned HALF_FRACTION_BITS;
  static const uint64_t PRECISION_64;

public:
  static const double TRUNCATE_PRECISION;

  static double truncate (const double d) {
    return (double (int64_t (d * TRUNCATE_PRECISION)) / TRUNCATE_PRECISION);
  }

  static Vec2 truncate (const Vec2& v) {
    return Vec2 (truncate (v.getX ()), truncate (v.getY ()));
  }


  static bool almostEqual (const double d1, const double d2) {
    return (fabs (d1 - d2) < EPSILON);
  }

  static bool almostEqual (const Vec2& v1, const Vec2& v2) {
    return almostEqual (v1.getX (), v2.getX ()) && almostEqual (v1.getY (), v2.getY ());
  }


  // It can be shown for basic operations that after truncation, the 
  // relative error remains within a small constant factor of TRUNCATE_PRECISION
  static inline double FP_ERR_LIM () {
    return 128.0 / (TRUNCATE_PRECISION);
  } 

  //! checks relative error
  static bool checkError (const double original, const double measured, bool useAssert=true) {

    double err = fabs (measured - original);
    if (original != 0.0) {
      err = fabs ((measured - original) / original);
    }

    bool withinLim = err < FP_ERR_LIM ();


    if (!withinLim) {
      fprintf (stderr, "WARNING: FPutils::checkError, relative error=%10.20g above limit, FP_ERR_LIM=%10.20g\n"
          , err, FP_ERR_LIM ());
    }

    if (useAssert) {
      assert ( withinLim && "Relative Error above limit?");
    }


    return withinLim;
  }


  static bool checkError (const Vec2& original, const Vec2& measured, bool useAssert=true) {

    return checkError (original.getX (), measured.getX (), useAssert) 
      && checkError (original.getY (), measured.getY (), useAssert);
  }

};

#endif // _FP_UTILS_H_
