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

#pragma once
#include "cuda/cuda_mtypes.h"

struct pr_CUDA_Context;

/* infrastructure */
struct pr_CUDA_Context* get_CUDA_context(int id);
bool init_CUDA_context(struct pr_CUDA_Context* ctx, int device);
void load_graph_CUDA(struct pr_CUDA_Context* ctx, MarshalGraph& g);
void test_cuda(struct pr_CUDA_Context* ctx);

/* application */
void initialize_graph_cuda(struct pr_CUDA_Context* ctx);
void test_graph_cuda(struct pr_CUDA_Context* ctx);
void pagerank_cuda(struct pr_CUDA_Context* ctx);

/* application messages */
void set_PRNode_pr_CUDA(struct pr_CUDA_Context* ctx, unsigned LID, float v);
float get_PRNode_pr_CUDA(struct pr_CUDA_Context* ctx, unsigned LID);

void set_PRNode_nout_CUDA(struct pr_CUDA_Context* ctx, unsigned LID,
                          unsigned nout);
unsigned get_PRNode_nout_CUDA(struct pr_CUDA_Context* ctx, unsigned LID);
void set_PRNode_nout_plus_CUDA(struct pr_CUDA_Context* ctx, unsigned LID,
                               unsigned nout);
