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
#include <dai/factorgraph.h>

using namespace std;
using namespace dai;

/* Input Arguments */

#define FILENAME_IN prhs[0]
#define NR_IN 1

/* Output Arguments */

#define PSI_OUT plhs[0]
#define NR_OUT 1

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  char* filename;

  // Check for proper number of arguments
  if ((nrhs != NR_IN) || (nlhs != NR_OUT)) {
    mexErrMsgTxt(
        "Usage: [psi] = dai_readfg(filename);\n\n"
        "\n"
        "INPUT:  filename   = filename of a .fg file\n"
        "\n"
        "OUTPUT: psi        = linear cell array containing the factors\n"
        "                     (psi{i} is a structure with a Member field\n"
        "                     and a P field, like a CPTAB).\n");
  }

  // Get input parameters
  size_t buflen;
  buflen   = mxGetN(FILENAME_IN) + 1;
  filename = (char*)mxCalloc(buflen, sizeof(char));
  mxGetString(FILENAME_IN, filename, buflen);

  // Read factorgraph
  FactorGraph fg;
  try {
    fg.ReadFromFile(filename);
  } catch (std::exception& e) {
    mexErrMsgTxt(e.what());
  }

  // Save factors
  vector<Factor> psi;
  for (size_t I = 0; I < fg.nrFactors(); I++)
    psi.push_back(fg.factor(I));

  // Hand over results to MATLAB
  PSI_OUT = Factors2mx(psi);

  return;
}
