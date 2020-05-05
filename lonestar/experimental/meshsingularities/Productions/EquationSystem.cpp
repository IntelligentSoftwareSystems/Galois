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

#include "EquationSystem.h"
#include <string>
#include <cmath>

#ifdef WITH_BLAS
extern "C" {
#include <cblas.h>
#include <clapack.h>
}
#endif

EquationSystem::EquationSystem(unsigned long n) {
  this->n = n;
  unsigned long i;

  // we are working on continuous area of memory

  matrix    = new double*[n];
  matrix[0] = new double[n * (n + 1)]();
  for (i = 0; i < n; ++i) {
    matrix[i] = matrix[0] + i * n;
  }

  if (matrix == NULL || matrix[0] == NULL) {
    throw std::string("Cannot allocate memory!");
  }

  rhs = matrix[0] + n * n;

  origPtr = matrix[0];
}

EquationSystem::EquationSystem(double** matrix, double* rhs,
                               unsigned long size) {
  this->n = size;
  unsigned long i;

  // we are working on continuous area of memory

  this->matrix    = new double*[n];
  this->matrix[0] = new double[n * (n + 1)]();

  for (i = 1; i < n; ++i) {
    this->matrix[i] = this->matrix[0] + i * n;
  }

  if (matrix == NULL || matrix[0] == NULL) {
    throw std::string("Cannot allocate memory!");
  }

  this->rhs = this->matrix[0] + n * n;

  for (i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      this->matrix[i][j] = matrix[i][j];
    }
    this->rhs[i] = rhs[i];
  }

  origPtr = this->matrix[0];
}

EquationSystem::~EquationSystem() {
  if (matrix != NULL) {
    delete[] origPtr;
    delete[] matrix;
  }
}

void EquationSystem::eliminate(const int rows) {
#ifdef WITH_BLAS
  //    int clapack_dgetrf(const enum CBLAS_ORDER Order, const int M, const int
  //    N,
  //                       double *A, const int lda, int *ipiv);
  int ipiv[rows];
  const int m = rows;
  const int k = n - rows;
  int error   = 0;
  error       = clapack_dgetrf(CblasRowMajor,
                         m, // size
                         m,
                         matrix[0], // pointer to data
                         n,         // LDA = matrix_size
                         ipiv);     // pivot vector

  if (error != 0) {
    printf("DGETRF error: %d\n", error);
  }

  // int clapack_dgetrs
  //   (const enum CBLAS_ORDER Order,
  //    const enum CBLAS_TRANSPOSE Trans,
  //    const int N,
  //    const int NRHS,
  //    const double *A,
  //    const int lda,
  //    const int *ipiv,
  //    double *B,
  //    const int ldb);

  error = clapack_dgetrs(CblasRowMajor, CblasNoTrans, m, k, matrix[0], n, ipiv,
                         matrix[0] + m, n);

  if (error != 0) {
    printf("DGETRS error: %d\n", error);
  }

  error = clapack_dgetrs(CblasRowMajor, CblasNoTrans, m, 1, matrix[0], n, ipiv,
                         rhs, n);

  if (error != 0) {
    printf("DGETRS error: %d\n", error);
  }

  // void cblas_dgemm(const enum CBLAS_ORDER Order,
  //                 const enum CBLAS_TRANSPOSE TransA,
  //                 const enum CBLAS_TRANSPOSE TransB,
  //                 const int M,
  //                 const int N,
  //                 const int K,
  //                 const double alpha,
  //                 const double *A,
  //                 const int lda,
  //                 const double *B,
  //                 const int ldb,
  //                 const double beta,
  //                 double *C,
  //                 const int ldc);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, k, k, m, -1.0,
              matrix[m], // C
              n,
              matrix[0] + m, // B
              n, 1.0,
              matrix[m] + m, // D
              n);

  // void cblas_dgemv(const enum CBLAS_ORDER Order,
  //                 const enum CBLAS_TRANSPOSE TransA,
  //                 const int M,
  //                 const int N,
  //                 const double alpha,
  //                 const double *A,
  //                 const int lda,
  //                 const double *X,
  //                 const int incX,
  //                 const double beta,
  //                 double *Y,
  //                 const int incY);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, k, m, -1.0, matrix[m], n, rhs, 1,
              1.0, rhs + m, 1);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      matrix[i][j] = i == j ? 1.0 : 0.0;
    }
  }

  for (int i = m; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      matrix[i][j] = 0.0;
    }
  }

#else
  double maxX;
  register int maxRow;
  double x;

  for (int i = 0; i < rows; ++i) {
    maxX   = fabs(matrix[i][i]);
    maxRow = i;

    for (int k = i + 1; k < rows; ++k) {
      if (fabs(matrix[k][i]) > maxX) {
        maxX   = fabs(matrix[k][i]);
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
      matrix[i][j] /= x;
    }

    rhs[i] /= x;
    for (int j = i + 1; j < n; ++j) {
      x = matrix[j][i];

      for (int k = i + 1; k < n; ++k) {
        matrix[j][k] -= x * matrix[i][k];
      }
      // xyz
      // matrix[j][i] = 0;
      rhs[j] -= x * rhs[i];
    }
  }
#endif
}

void EquationSystem::backwardSubstitute(const int startingRow) {
  for (int i = startingRow; i >= 0; --i) {
    double sum = rhs[i];
    for (int j = n - 1; j >= i + 1; --j) {
      sum -= matrix[i][j] * rhs[j];
      matrix[i][j] = 0.0;
    }
    rhs[i] = sum; // / matrix[i][i]; // after elimination we have always 1.0 at
                  // matrix[i][i]
                  // do not need to divide by matrix[i][i]
  }
}

void EquationSystem::swapCols(const int i, const int j) {
  for (int k = 0; k < n; ++k) {
    double tmp   = matrix[k][i];
    matrix[k][i] = matrix[k][j];
    matrix[k][j] = tmp;
  }
}

void EquationSystem::swapRows(const int i, const int j) {
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
