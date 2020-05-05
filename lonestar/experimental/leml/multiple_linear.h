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

#ifndef MULTIPLE_LINEAR_H
#define MULTIPLE_LINEAR_H

#include "smat.h"

/*
 *  W = argmin_{W}  sum_{ij} C_{ij} * loss(Y,  X*W')_{ij} +  0.5*||W||^2
 *
 *  Y: m*n sparse matrix
 *  X: m*k dense matrix in row-majored
 *  W: n*k dense matrix in row-majored
 *
 *  if Y is fully observed
 *     C_{ij} = Y_{ij} is observed ? Cp : Cn
 *  if Y is partially observed
 *     C_{ij} = Y_{ij} > 0 Cp : Cn
 *
 *  loss:
 *    ls: squared loss = |Y_{ij} - XW'_{ij}|^2
 *    lr: logistic loss
 *
 * */

class multiple_linear_problem {
public:
  smat_t* Y;
  double* X;
  int k;
  multiple_linear_problem(smat_t* Y, double* X, int k) {
    this->Y = Y;
    this->X = X;
    this->k = k;
  }
};

class multiple_linear_parameter {
public:
  double Cp, Cn;
  int max_tron_iter, max_cg_iter;
  double eps;
  int verbose;
  multiple_linear_parameter(double Cp = 0.1, double Cn = 0.1,
                            int max_tron_iter = 5, int max_cg_iter = 20,
                            double eps = 0.1, int verbose = 1) {
    this->Cp            = Cp;
    this->Cn            = Cn;
    this->max_tron_iter = max_tron_iter;
    this->max_cg_iter   = max_cg_iter;
    this->eps           = eps;
    this->verbose       = verbose;
  }
};

// Squared loss with Cholesky factorization as the solver
int multiple_l2r_ls_chol(multiple_linear_problem* prob,
                         multiple_linear_parameter* param, double* W);
// Squared loss with Cholesky factorization as the solver for fully observed
// label matrix
int multiple_l2r_ls_chol_full(multiple_linear_problem* prob,
                              multiple_linear_parameter* param, double* W);
// Squared loss with Cholesky factorization as the solver for fully observed
// label matrix with weight support
int multiple_l2r_ls_chol_full_weight(multiple_linear_problem* prob,
                                     multiple_linear_parameter* param,
                                     double* W);
// Squared loss with TRON as the solver
int multiple_l2r_ls_tron(multiple_linear_problem* prob,
                         multiple_linear_parameter* param, double* W);
// logistic loss with TRON as the solver
int multiple_l2r_lr_tron(multiple_linear_problem* prob,
                         multiple_linear_parameter* param, double* W);
// Square hinge loss with TRON as the solver
int multiple_l2r_l2svc_tron(multiple_linear_problem* prob,
                            multiple_linear_parameter* param, double* W);

#endif
