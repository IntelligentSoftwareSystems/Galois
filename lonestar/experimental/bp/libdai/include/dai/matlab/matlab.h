/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
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
mxArray *Factors2mx(const std::vector<Factor> &Ps);

/// Convert cell vector of CPTAB-like structs to vector<Factor>
std::vector<Factor> mx2Factors(const mxArray *psi, long verbose);

/// Convert CPTAB-like struct to Factor
Factor mx2Factor(const mxArray *psi);


} // end of namespace dai


#endif
