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
