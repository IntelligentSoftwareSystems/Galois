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

/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found
 * in the LICENSE file.
 */

#include <iostream>
#include "mex.h"
#include <dai/matlab/matlab.h>
#include <dai/factor.h>

using namespace std;
using namespace dai;

/* Input Arguments */

#define PSI_IN prhs[0]
#define I_IN prhs[1]
#define J_IN prhs[2]
#define NR_IN 3

/* Output Arguments */

#define N_OUT plhs[0]
#define NR_OUT 1

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  size_t ilabel, jlabel;

  // Check for proper number of arguments
  if ((nrhs != NR_IN) || (nlhs != NR_OUT)) {
    mexErrMsgTxt("Usage: N = dai_potstrength(psi,i,j);\n\n"
                 "\n"
                 "INPUT:  psi        = structure with a Member field and a P "
                 "field, like a CPTAB.\n"
                 "        i          = label of a variable in psi.\n"
                 "        j          = label of another variable in psi.\n"
                 "\n"
                 "OUTPUT: N          = strength of psi in direction i->j.\n");
  }

  // Get input parameters
  Factor psi = mx2Factor(PSI_IN);
  ilabel     = (size_t)*mxGetPr(I_IN);
  jlabel     = (size_t)*mxGetPr(J_IN);

  // Find variable in psi with label ilabel
  Var i;
  for (VarSet::const_iterator n = psi.vars().begin(); n != psi.vars().end();
       n++)
    if (n->label() == ilabel) {
      i = *n;
      break;
    }
  DAI_ASSERT(i.label() == ilabel);

  // Find variable in psi with label jlabel
  Var j;
  for (VarSet::const_iterator n = psi.vars().begin(); n != psi.vars().end();
       n++)
    if (n->label() == jlabel) {
      j = *n;
      break;
    }
  DAI_ASSERT(j.label() == jlabel);

  // Calculate N(psi,i,j);
  double N = psi.strength(i, j);

  // Hand over result to MATLAB
  N_OUT             = mxCreateDoubleMatrix(1, 1, mxREAL);
  *(mxGetPr(N_OUT)) = N;

  return;
}
