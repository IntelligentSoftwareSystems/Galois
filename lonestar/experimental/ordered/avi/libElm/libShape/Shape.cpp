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
 * Shape.cpp
 * DG++
 *
 * Created by Adrian Lew on 9/7/06.
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

#include "Shape.h"
#include <cmath>
#include <iostream>

bool Shape::consistencyTest(const double* X, const double Pert) const {
  double* DValNum  = new double[getNumFunctions() * getNumVariables()];
  double* DValAnal = new double[getNumFunctions() * getNumVariables()];
  double* Xpert    = new double[getNumVariables()];
  double* Valplus  = new double[getNumFunctions()];
  double* Valminus = new double[getNumFunctions()];

  if (Pert <= 0)
    std::cerr
        << "Shape::ConsistencyTest - Pert cannot be less or equal than zero\n";

  for (size_t i = 0; i < getNumVariables(); i++) {
    Xpert[i] = X[i];
    for (size_t a = 0; a < getNumFunctions(); a++)
      DValAnal[a * getNumVariables() + i] = getDVal(a, X, i);
  }

  for (size_t i = 0; i < getNumVariables(); i++) {
    Xpert[i] = X[i] + Pert;
    for (size_t a = 0; a < getNumFunctions(); a++)
      Valplus[a] = getVal(a, Xpert);

    Xpert[i] = X[i] - Pert;
    for (size_t a = 0; a < getNumFunctions(); a++)
      Valminus[a] = getVal(a, Xpert);

    Xpert[i] = X[i];

    for (size_t a = 0; a < getNumFunctions(); a++)
      DValNum[a * getNumVariables() + i] =
          (Valplus[a] - Valminus[a]) / (2 * Pert);
  }

  double error        = 0;
  double normX        = 0;
  double normDValNum  = 0;
  double normDValAnal = 0;

  for (size_t i = 0; i < getNumVariables(); i++) {
    normX += X[i] * X[i];

    for (size_t a = 0; a < getNumFunctions(); a++) {
      error += pow(DValAnal[a * getNumVariables() + i] -
                       DValNum[a * getNumVariables() + i],
                   2.);
      normDValAnal += pow(DValAnal[a * getNumVariables() + i], 2.);
      normDValNum += pow(DValNum[a * getNumVariables() + i], 2.);
    }
  }
  error        = sqrt(error);
  normX        = sqrt(normX);
  normDValAnal = sqrt(normDValAnal);
  normDValNum  = sqrt(normDValNum);

  delete[] Valplus;
  delete[] Valminus;
  delete[] Xpert;
  delete[] DValNum;
  delete[] DValAnal;

  if (error * (normX + Pert) <
      (normDValAnal < normDValNum ? normDValNum : normDValAnal) * Pert * 10) {
    return true;
  } else {
    return false;
  }
}
