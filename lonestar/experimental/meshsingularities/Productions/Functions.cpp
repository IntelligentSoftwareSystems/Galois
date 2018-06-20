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
 * Functions.cpp
 *
 *  Created on: Aug 22, 2013
 *      Author: kjopek
 */

#include "Functions.h"
#include <cstdarg>

const Function functionsTable[] = {{"f1(x, [y, [z]]) = 1.0", f1},
                                   {"f2(x, [y, [z]]) = x+[y+[z]]", f2},
                                   {"f3(x, [y, [z]]) = x^2+[y^2+[z^2]]", f3},
                                   {"f4(x, [y, [z]]) = x^3+[y^3+[z^3]]", f4}};

double f1(int dim, ...) { return 1.0; }

double f2(int dim, ...) {
  double* data  = new double[dim];
  double result = 0;
  va_list args;

  va_start(args, dim);
  for (int i = 0; i < dim; ++i) {
    data[i] = va_arg(args, double);
  }
  va_end(args);

  if (dim == 1) {
    result = data[0];
  } else if (dim == 2) {
    result = data[0] + data[1];
  } else if (dim == 3) {
    result = data[0] + data[1] + data[2];
  }

  delete[] data;
  return result;
}

double f3(int dim, ...) {
  double* data  = new double[dim];
  double result = 0;
  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim; ++i) {
    data[i] = va_arg(args, double);
  }
  va_end(args);

  if (dim == 1) {
    result = data[0] * data[0];
  } else if (dim == 2) {
    result = data[0] * data[0] + data[1] * data[1];
  } else if (dim == 3) {
    result = data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
  }

  delete[] data;
  return result;
}

double f4(int dim, ...) {
  double* data  = new double[dim];
  double result = 0;
  va_list args;
  va_start(args, dim);
  for (int i = 0; i < dim; ++i) {
    data[i] = va_arg(args, double);
  }
  va_end(args);

  if (dim == 1) {
    result = data[0] * data[0] * data[0];
  } else if (dim == 2) {
    result = data[0] * data[0] * data[0] + data[1] * data[1] * data[1];
  } else if (dim == 3) {
    result = data[0] * data[0] * data[0] + data[1] * data[1] * data[1] +
             data[2] * data[2] * data[2];
  }

  delete[] data;
  return result;
}
