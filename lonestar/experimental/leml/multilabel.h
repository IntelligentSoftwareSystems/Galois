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

#ifndef MULTILABEL_H
#define MULTILABEL_H

#include "bilinear.h"

enum { ML_LS, ML_LR, ML_L2SVC };

class multilabel_problem {
public:
  bilinear_problem* training_set;
  bilinear_problem* test_set;
  multilabel_problem(bilinear_problem* training_set,
                     bilinear_problem* test_set) {
    this->training_set = training_set;
    this->test_set     = test_set;
  }
};

class multilabel_parameter : public bilinear_parameter {
public:
  int maxiter;
  int top_p;
  int k;
  int threads;
  int reweighting;
  bool predict;
  // Parameters for Wsabie
  double lrate; // learning rate for wsabie
  multilabel_parameter() {
    bilinear_parameter();
    reweighting = 0;
    maxiter     = 10;
    top_p       = 20;
    k           = 10;
    threads     = 8;
    lrate       = 0.01;
    predict     = true;
  }
};

#ifdef __cplusplus
extern "C" {
#endif

void multilabel_train(multilabel_problem* prob, multilabel_parameter* param,
                      double* W, double* H);

#ifdef __cplusplus
}
#endif

#endif
