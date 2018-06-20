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

#include "EquationSystem.h"
#include <string>
#include <cmath>
#include <cstdlib>

extern "C" {
#include <cblas.h>
#include <clapack.h>
}

EquationSystem::EquationSystem(unsigned long n, SolverMode mode) {
  this->n    = n;
  this->mode = mode;
  unsigned long i;

  // we are working on continuous area of memory
  posix_memalign((void**)&matrix, sizeof(double*), n * sizeof(double*));
  posix_memalign((void**)&matrix[0], sizeof(double),
                 n * (n + 1) * sizeof(double));

  for (i = 0; i < n; ++i) {
    matrix[i] = matrix[0] + i * n;
  }

  if (matrix == NULL || matrix[0] == NULL) {
    throw std::string("Cannot allocate memory!");
  }

  rhs = matrix[0] + n * n;

  origPtr = matrix[0];
}

EquationSystem::~EquationSystem() {
  if (matrix != NULL) {
    free((void*)origPtr);
    free((void*)matrix);
  }
}

int EquationSystem::eliminate(const int rows) {

  int error = 0;

  const int m = rows;
  const int k = n - rows;

  if (mode == LU) {
    int ipiv[m];

    error = clapack_dgetrf(CblasColMajor,
                           m, // size
                           m,
                           matrix[0], // pointer to data
                           n,         // LDA = matrix_size
                           ipiv);     // pivot vector

    if (error != 0) {
      printf("DGETRF error: %d\n", error);
      return error;
    }

    error = clapack_dgetrs(CblasColMajor, CblasNoTrans, m, k, matrix[0], n,
                           ipiv, matrix[m], n);

    if (error != 0) {
      printf("DGETRS error: %d\n", error);
      return error;
    }

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, k, k, m, -1.0,
                matrix[0] + m, n, matrix[m], n, 1.0,
                matrix[m] + m, // D
                n);
  } else if (mode == CHOLESKY) {
    error = clapack_dpotrf(CblasColMajor, CblasUpper,
                           m,         // size
                           matrix[0], // pointer to data
                           n          // LDA = matrix_size
    );                                // pivot vector

    if (error != 0) {
      printf("DPOTRF error: %d\n", error);
      return error;
    }

    clapack_dpotrs(CblasColMajor, CblasUpper, m, k, matrix[0], n, matrix[m], n);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, k, k, m, -1.0,
                matrix[0] + m, n,
                matrix[m], // B
                n, 1.0,
                matrix[m] + m, // D
                n);
  } else {
    double maxX;
    register int maxRow;
    double x;

    for (int i = 0; i < rows; ++i) {
      maxX   = fabs(matrix[i][i]);
      maxRow = i;

      for (int k = i + 1; k < rows; ++k) {
        if (fabs(matrix[i][k]) > maxX) {
          maxX   = fabs(matrix[i][k]);
          maxRow = k;
        }
      }

      if (maxRow != i) {
        swapRows(i, maxRow);
      }

      x = matrix[i][i];
      // on diagonal - only 1.0
      matrix[i][i] = 1.0;

      for (int j = i + 1; j < n; ++j) {
        matrix[j][i] /= x;
      }

      rhs[i] /= x;
      for (int j = i + 1; j < n; ++j) {
        x = matrix[i][j];

        for (int k = i + 1; k < n; ++k) {
          matrix[k][j] -= x * matrix[k][i];
        }
        // xyz
        // matrix[j][i] = 0;
        rhs[j] -= x * rhs[i];
      }
    }
  }
  return 0;
}

void EquationSystem::backwardSubstitute(const int startingRow) {
  if (mode == OLD) {
    for (int i = startingRow; i >= 0; --i) {
      double sum = rhs[i];
      for (int j = n - 1; j >= i + 1; --j) {
        sum -= matrix[i][j] * rhs[j];
        matrix[i][j] = 0.0;
      }
      rhs[i] = sum; // / matrix[i][i]; // after elimination we have always 1.0
                    // at matrix[i][i]
                    // do not need to divide by matrix[i][i]
    }
  }
}

void EquationSystem::swapRows(const int i, const int j) {
  for (int k = 0; k < n; ++k) {
    double tmp   = matrix[k][i];
    matrix[k][i] = matrix[k][j];
    matrix[k][j] = tmp;
  }
}

void EquationSystem::swapCols(const int i, const int j) {
  // reduced complexity from O(n) to O(1)
  double tmp;
  double* tmpPtr = matrix[i];
  matrix[i]      = matrix[j];
  matrix[j]      = tmpPtr;

  tmp    = rhs[i];
  rhs[i] = rhs[j];
  rhs[j] = tmp;
}

void EquationSystem::checkRow(int row_nr, int* values, int values_cnt) {
  double v = 0;
  for (int i = 0; i < values_cnt; i++)
    v += matrix[row_nr][values[i]];

  printf("DIFF : %lf\n", rhs[row_nr] - v);
}

void EquationSystem::print() const {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::printf("% .6f ", matrix[i][j]);
    }
    std::printf(" | % .6f\n", rhs[i]);
  }
}
