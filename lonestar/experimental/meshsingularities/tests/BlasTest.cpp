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
 * BlasTest.cpp
 *
 *  Example of usage of BLAS/LAPACK to compute
 *  Schur complement:
 *
 *  Ax + By = b_1
 *  Cx + Dy = b_2
 *
 *  LU(A)x + A^{-1}By = b_1
 *  Cx + (D-CA^{-1}B)y = b_2
 *
 * LU(A) - DGETRF
 * A^{-1}B - DGETRS
 * D-CA^{-1}B - DGETRS
 *
 * Also if C = B^T and A, D - symmetric, positive definite we can:
 * LL(A)x + A^{-1}By = b_1
 * B^T x + (D-B^TA^{-1}B)y = b_2
 *
 * LL(A) - DPOSRF
 * A^{-1}B - DPOSRS
 * D-CA^{-1}B - DPOSRS
 *
 */

#include <cstdio>
#include <cmath>
using namespace std;

extern "C" {
#include "cblas.h"
#include "clapack.h"
}

void test_lu_fact() {
  int error = 0;

  const int n = 4;
  const int m = 2;
  const int k = 2;
  int ipiv[m] = {0, 0};

  double** matrix;
  double* rhs;

  double data[4][4] = {
      {1, 1, 0, 3}, {2, 1, -1, 1}, {3, -1, -1, 2}, {2, 2, 3, -1}};

  double data_r[4] = {4, 1, -3, 4};

  matrix    = new double*[n];
  matrix[0] = new double[n * (n + 1)];
  matrix[1] = matrix[0] + n;
  matrix[2] = matrix[0] + 2 * n;
  matrix[3] = matrix[0] + 3 * n;
  rhs       = matrix[0] + 4 * n;

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) { // col-major!
      matrix[i][j] = data[j][i];
    }
    rhs[i] = data_r[i];
  }

  printf("Input matrix is: \n");
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      printf("%8.3lf ", matrix[j][i]);
    }
    printf(" | %8.3lf\n", rhs[i]);
  }

  error = clapack_dgetrf(CblasColMajor,
                         m, // size
                         m,
                         matrix[0], // pointer to data
                         n,         // LDA = matrix_size
                         ipiv);     // pivot vector

  if (!error) {
    printf("DGETRF error: %d\n", error);
  }

  printf("after DGETRF matrix is: \n");
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      printf("%8.3lf ", matrix[j][i]);
    }
    printf(" | %8.3lf\n", rhs[i]);
  }

  printf("IPIV vector: ");
  for (int i = 0; i < m; ++i) {
    printf("%d ", ipiv[i]);
  }
  printf("\n");

  clapack_dgetrs(CblasColMajor, CblasNoTrans, m, k, matrix[0], n, ipiv,
                 matrix[m], n);

  printf("after DGETRS matrix is: \n");
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      printf("%8.3lf ", matrix[j][i]);
    }
    printf(" | %8.3lf\n", rhs[i]);
  }

  printf("IPIV vector: ");
  for (int i = 0; i < m; ++i) {
    printf("%d ", ipiv[i]);
  }
  printf("\n");

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
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, k, k, m, -1.0,
              matrix[0] + m, // C
              n,
              matrix[m], // B
              n, 1.0,
              matrix[m] + m, // D
              n);
  printf("after DGEMM matrix is: \n");
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      printf("%8.3lf ", matrix[j][i]);
    }
    printf(" | %8.3lf\n", rhs[i]);
  }

  printf("IPIV vector: ");
  for (int i = 0; i < m; ++i) {
    printf("%d ", ipiv[i]);
  }
  printf("\n");

  delete[] matrix[0];
  delete[] matrix;
}

void test_ll_fact() {
  int error = 0;

  const int n = 4;
  const int m = 2;
  const int k = 2;

  double** matrix;
  double* rhs;
  double data[4][4] = {{2, 1, 3, 2}, {1, 2, 2, 3}, {3, 2, 9, 7}, {2, 3, 7, 9}};

  double data_r[4] = {4, 1, -3, 4};

  matrix    = new double*[n];
  matrix[0] = new double[n * (n + 1)];
  matrix[1] = matrix[0] + n;
  matrix[2] = matrix[0] + 2 * n;
  matrix[3] = matrix[0] + 3 * n;
  rhs       = matrix[0] + 4 * n;

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) { // col-major!
      matrix[i][j] = data[j][i];
    }
    rhs[i] = data_r[i];
  }

  printf("Input matrix is: \n");
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      printf("%8.3lf ", matrix[j][i]);
    }
    printf(" | %8.3lf\n", rhs[i]);
  }

  error = clapack_dpotrf(CblasColMajor, CblasUpper,
                         m,         // size
                         matrix[0], // pointer to data
                         n          // LDA = matrix_size
  );                                // pivot vector

  if (!error) {
    printf("DPOTRF error: %d\n", error);
  }

  printf("after DPOTRF matrix is: \n");
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      printf("%8.3lf ", matrix[j][i]);
    }
    printf(" | %8.3lf\n", rhs[i]);
  }

  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, m,
              k, 1.0, matrix[0], n, matrix[m], n);
  printf("after DTRSM matrix is: \n");
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      printf("%8.3lf ", matrix[j][i]);
    }
    printf(" | %8.3lf\n", rhs[i]);
  }

  cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, k, m, -1.0,
              matrix[m], // L**(-1) * B
              n, 1.0,
              matrix[m] + m, // D
              n);
  printf("after DSYRK matrix is: \n");
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      printf("%8.3lf ", matrix[j][i]);
    }
    printf(" | %8.3lf\n", rhs[i]);
  }

  delete[] matrix[0];
  delete[] matrix;
}

int main() {
  printf("\n==== LU FACTORIZATION ====\n");
  test_lu_fact();
  printf("\n==== LL* FACTORIZATION ====\n");
  test_ll_fact();

  return 0;
}
