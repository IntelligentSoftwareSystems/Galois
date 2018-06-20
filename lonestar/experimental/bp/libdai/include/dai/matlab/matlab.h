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

/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found
 * in the LICENSE file.
 */

/// \file
/// \brief Defines some utility functions for interfacing with MatLab

#ifndef __defined_libdai_matlab_h
#define __defined_libdai_matlab_h

#include "mex.h"
#include <dai/factor.h>

namespace dai {

#ifdef SMALLMEM
typedef int mwSize;
typedef int mwIndex;
#endif

/// Convert vector<Factor> structure to a cell vector of CPTAB-like structs
mxArray* Factors2mx(const std::vector<Factor>& Ps);

/// Convert cell vector of CPTAB-like structs to vector<Factor>
std::vector<Factor> mx2Factors(const mxArray* psi, long verbose);

/// Convert CPTAB-like struct to Factor
Factor mx2Factor(const mxArray* psi);

} // end of namespace dai

#endif
