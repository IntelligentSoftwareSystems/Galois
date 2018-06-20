/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

/*
 * AuxDefs.h: Common definitions
 * DG++
 *
 * Created by Adrian Lew on 9/4/06.
 *
 * Copyright (c) 2006 Adrian Lew
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef _AUXDEFS_H_
#define _AUXDEFS_H_

#include <cstddef>
#include <cmath>
#include <cstdlib>

#include <vector>

#include "galois/gstl.h"

//! Nodal indices, starting at 0. : amber
typedef size_t GlobalNodalIndex;

//! Degree of freedom indices, starting at 0.
typedef size_t GlobalDofIndex;

//! Element indices, starting at 0.
typedef size_t GlobalElementIndex;

//! commonly used vector and vector<vector>
typedef galois::gstl::Vector<double> VecDouble;
typedef galois::gstl::Vector<galois::gstl::Vector<double>> MatDouble;
typedef galois::gstl::Vector<
    galois::gstl::Vector<galois::gstl::Vector<galois::gstl::Vector<double>>>>
    FourDVecDouble;

typedef galois::gstl::Vector<bool> VecBool;
typedef galois::gstl::Vector<galois::gstl::Vector<bool>> MatBool;

using VecSize_t = galois::gstl::Vector<size_t>;

//! constants
const double TOLERANCE = 1e-20;

struct DoubleComparator {
  static inline int compare(double left, double right) {
    double tdiff = left - right;

    if (fabs(tdiff) < TOLERANCE) {
      return 0;

    } else if (tdiff > 0.0) {
      return 1;

    } else if (tdiff < 0.0) {
      return -1;

    } else {
      abort(); // shouldn't reach here
      return 0;
    }
  }
};

#endif
