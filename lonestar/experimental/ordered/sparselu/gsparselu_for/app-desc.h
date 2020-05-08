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

/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de
 * Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify */
/*  it under the terms of the GNU General Public License as published by */
/*  the Free Software Foundation; either version 2 of the License, or */
/*  (at your option) any later version. */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful, */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/*  GNU General Public License for more details. */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License */
/*  along with this program; if not, write to the Free Software */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
 * USA            */
/**********************************************************************************************/

#include "omp-tasks-app.h"

#define BOTS_APP_NAME "SparseLU (For version)"
#define BOTS_APP_PARAMETERS_DESC "S1=%dx%d, S2=%dx%d"
#define BOTS_APP_PARAMETERS_LIST                                               \
  , bots_arg_size, bots_arg_size, bots_arg_size_1, bots_arg_size_1

#define BOTS_APP_USES_ARG_SIZE
#define BOTS_APP_DEF_ARG_SIZE 50
#define BOTS_APP_DESC_ARG_SIZE "Matrix Size"

#define BOTS_APP_USES_ARG_SIZE_1
#define BOTS_APP_DEF_ARG_SIZE_1 100
#define BOTS_APP_DESC_ARG_SIZE_1 "Submatrix Size"

#define BOTS_APP_USES_ARG_SIZE_2
#define BOTS_APP_DEF_ARG_SIZE_2 1
#define BOTS_APP_DESC_ARG_SIZE_2 "Number of Threads"

#define BOTS_APP_USES_ARG_FILE
#define BOTS_APP_DESC_ARG_FILE                                                 \
  "Graph file (for synthetic generator use empty string)"

#define BOTS_APP_USES_ARG_CUTOFF_1
#define BOTS_APP_DEF_ARG_CUTOFF_1 0
#define BOTS_APP_DESC_ARG_CUTOFF_1 "0: use bulk-synchronous >0: use ikdg "

#define BOTS_APP_INIT float **SEQ, **BENCH;

void sparselu_init(float*** pM, char* pass);
void sparselu_fini(float** M, char* pass);
void sparselu_seq_call(float** SEQ);
void sparselu_par_call(float** BENCH);
int sparselu_check(float** SEQ, float** BENCH);

#define KERNEL_INIT sparselu_init(&BENCH, "benchmark");
#define KERNEL_CALL sparselu_par_call(BENCH);
#define KERNEL_FINI sparselu_fini(BENCH, "benchmark");

#define KERNEL_SEQ_INIT sparselu_init(&SEQ, "serial");
#define KERNEL_SEQ_CALL sparselu_seq_call(SEQ);
#define KERNEL_SEQ_FINI sparselu_fini(SEQ, "serial");

#define BOTS_APP_CHECK_USES_SEQ_RESULT
#define KERNEL_CHECK sparselu_check(SEQ, BENCH);
