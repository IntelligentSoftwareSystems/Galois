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
#include <cstdint>

#include "fixed_point.h"
#include "GeomUtils.h"

using FP = fpml::fixed_point<int64_t, 24, 40>;

struct FPutils: private boost::noncopyable {

  static const FP EPSILON;
  static const FP ERROR_LIMIT;

  static FP sqrt (const FP& t) {
    double d = std::sqrt (double (t));
    assert (std::fabs ((double (t.sqrt ()) - d) / d) < 1e-5);

    return t.sqrt ();
  }

  static bool almostEqual (const FP& d1, const FP& d2) {
    return (fabs (d1 - d2) < EPSILON);
  }

  class Vec2;

  static bool almostEqual (const Vec2& v1, const Vec2& v2);


  //! checks relative error
  static bool checkError (const FP& original, const FP& measured, bool useAssert=true) {

    FP err = (measured - original).fabs ();;
    if (original != 0.0) {
      err = ((measured - original) / original).fabs ();
    }

    bool withinLim = err < ERROR_LIMIT;


    if (!withinLim) {
      fprintf (stderr, "WARNING: FPutils::checkError, relative error=%10.20g above limit, ERROR_LIMIT=%10.20g\n"
          , err, ERROR_LIMIT ());
    }

    if (useAssert) {
      assert ( withinLim && "Relative Error above limit?");
    }


    return withinLim;
  }


  static bool checkError (const Vec2& original, const Vec2& measured, bool useAssert=true);

  static const FP& truncate (const FP& val) { return val; }

  static const Vec2& truncate (const Vec2& vec) { return vec; }

};

#endif // _FP_UTILS_H_
