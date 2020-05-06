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

/*
 * NumaEquationSystem.cpp
 *
 *  Created on: Aug 30, 2013
 *      Author: kjopek
 */

#include "NumaEquationSystem.h"
#include <numa.h>
#include <string>

NumaEquationSystem::NumaEquationSystem(int n, int node) {
  if (numa_available()) {
    throw std::string("NUMA must be enabled to use NumaEquationSystem");
  }

  this->n = n;

  // we are working on continuous area of memory

  matrix    = (double**)numa_alloc_onnode(n * sizeof(double*), node);
  matrix[0] = (double*)numa_alloc_onnode(n * (n + 1) * sizeof(double), node);
  for (int i = 0; i < n; ++i) {
    matrix[i] = matrix[0] + i * n;
  }

  rhs = matrix[0] + n * n;
}

NumaEquationSystem::NumaEquationSystem(double** matrix, double* rhs, int size,
                                       int node) {
  if (!numa_available()) {
    throw std::string("NUMA must be enabled to use NumaEquationSystem");
  }

  this->n = n;

  // we are working on continuous area of memory

  this->matrix = (double**)numa_alloc_onnode(n * sizeof(double*), node);
  this->matrix[0] =
      (double*)numa_alloc_onnode(n * (n + 1) * sizeof(double), node);
  for (int i = 0; i < n; ++i) {
    this->matrix[i] = matrix[0] + i * n;
  }

  this->rhs = matrix[0] + n * n;

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      this->matrix[i][j] = matrix[i][j];
    }
    this->rhs[i] = rhs[i];
  }
}

NumaEquationSystem::~NumaEquationSystem() {
  numa_free((void*)matrix[0], n * (n + 1) * sizeof(double));
  numa_free((void*)matrix, n * sizeof(double*));

  matrix = NULL;
}
