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

#ifndef WSABIE_H
#define WSABIE_H

#include "bilinear.h"
#include "dmat.h"
#include "smat.h"
#include "multilabel.h"

#ifdef __cplusplus
extern "C" {
#endif

double get_wsabieC(const multilabel_parameter* param);

// Project
void wsabie_model_projection(double* W, double* H, long nr_feats,
                             long nr_labels, long k, double wsabieC);

// nr_samples = 0 => nr_samples = # postive labels in Y
void wsabie_updates(multilabel_problem* prob, multilabel_parameter* param,
                    double* W, double* H, long nr_samples = 0);
void wsabie_updates_new(multilabel_problem* prob, multilabel_parameter* param,
                        double* W, double* H, long nr_samples = 0);
void wsabie_updates_new2(multilabel_problem* prob, multilabel_parameter* param,
                         double* W, double* H, long nr_samples = 0);
void wsabie_updates_new3(multilabel_problem* prob, multilabel_parameter* param,
                         double* W, double* H, long nr_samples = 0);
void wsabie_updates_new4(multilabel_problem* prob, multilabel_parameter* param,
                         double* W, double* H, long nr_samples = 0);
void wsabie_updates_3(multilabel_problem* prob, multilabel_parameter* param,
                      double* H, long nr_samples = 0);
void wsabie_updates_2(multilabel_problem* prob, multilabel_parameter* param,
                      double* W, double* H, long* perm, int t0,
                      long nr_samples = 0);
void wsabie_updates_4(multilabel_problem* prob, multilabel_parameter* param,
                      double* H, long* perm, int t0, long nr_samples = 0);

#ifdef __cplusplus
}
#endif

#endif
