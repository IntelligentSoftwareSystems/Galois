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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <iostream>
#include <cstdio>

#include <cstdlib>
#include <algorithm>
#include <utility>
#include <map>
#include <queue>
#include <set>
#include <vector>

#include <omp.h>

extern "C" {
// int dposv_(char *uplo, int *n, int *nrhs, double *a, int *lda, double *b, int
// *ldb, int *info);
int dposv_(char* uplo, ptrdiff_t* n, ptrdiff_t* nrhs, double* a, ptrdiff_t* lda,
           double* b, ptrdiff_t* ldb, ptrdiff_t* info);

// C <= alpha*A*B + beta*C
// *trans = "T" or "N"
//
void dgemm_(char* transa, char* transb, ptrdiff_t* m, ptrdiff_t* n,
            ptrdiff_t* k, double* alpha, double* a, ptrdiff_t* lda, double* b,
            ptrdiff_t* ldb, double* beta, double* c, ptrdiff_t* ldc);
}

bool ls_solve_chol_matrix(double* A, int n, double* B, int m) {
  ptrdiff_t nn = n, lda = n, ldb = n, nrhs = m, info;
  // int nn=n, lda=n, ldb=n, nrhs=1, info;
  char uplo = 'U';
  dposv_(&uplo, &nn, &nrhs, A, &lda, B, &ldb, &info);
  return (info == 0);
}

// A, B, C are stored in column major!
void matrix_matrix(double alpha, double* A, bool trans_A, double* B,
                   bool trans_B, double beta, double* C, int m, int n, int k) {
  ptrdiff_t mm = m, nn = n, kk = k;
  ptrdiff_t lda = trans_A ? kk : mm, ldb = trans_B ? nn : kk, ldc = m;
  char transpose = 'T', notranspose = 'N';
  char* transa = trans_A ? &transpose : &notranspose;
  char* transb = trans_B ? &transpose : &notranspose;
  dgemm_(transa, transb, &mm, &nn, &kk, &alpha, A, &lda, B, &ldb, &beta, C,
         &ldc);
}

// Input: an n*k row-major matrix H
// Output: an k*k matrix H^TH
void doHTH(double* H, double* HTH, int n, int k) {
  bool transpose = true;
  matrix_matrix(1.0, H, !transpose, H, transpose, 0.0, HTH, k, k, n);
}
// Input: an n*k row-major matrix V and an k*k row-major symmetric matrix M
// Output: an n*k row-major matrix MV = alpha*V*M + beta MV
void doVM(double alpha, double* V, double* M, double beta, double* VM, int n,
          int k) {
  bool transpose = true;
  matrix_matrix(alpha, M, !transpose, V, !transpose, beta, VM, k, n, k);
}

int main() {
  int k       = 100;
  int m       = 10000;
  double* H   = (double*)malloc(sizeof(double) * k * m);
  double* HTH = (double*)malloc(sizeof(double) * k * k);
  for (int i = 0; i < k; i++)
    for (int j = 0; j < m; j++)
      H[i * m + j] = 5.0;
  omp_set_num_threads(8);
  doHTH(H, HTH, m, k);
  double sum = 0;
  for (int t = 0; t < k * k; t++)
    sum += HTH[t];
  printf("sum %.7g real %.7g diff %.7g\n", sum, k * k * m * 25.0,
         k * k * m * 25.0 - sum);
}
