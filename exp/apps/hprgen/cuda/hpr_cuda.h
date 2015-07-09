#pragma once
#include "cuda/cuda_mtypes.h"

struct pr_CUDA_Context;

/* infrastructure */
struct pr_CUDA_Context *get_CUDA_context(int id);
bool init_CUDA_context(struct pr_CUDA_Context *ctx, int device);
void load_graph_CUDA(struct pr_CUDA_Context *ctx, MarshalGraph &g);
void test_cuda(struct pr_CUDA_Context *ctx);

/* application */
void initialize_graph_cuda(struct pr_CUDA_Context *ctx);
void test_graph_cuda(struct pr_CUDA_Context *ctx);
void pagerank_cuda(struct pr_CUDA_Context *ctx);

/* application messages */
void set_pr_CUDA(struct pr_CUDA_Context *ctx, unsigned LID, float v);
float get_pr_CUDA(struct pr_CUDA_Context *ctx, unsigned LID);

void set_nout_CUDA(struct pr_CUDA_Context *ctx, unsigned LID, unsigned nout);
unsigned getNodeAttr_CUDA(struct pr_CUDA_Context *ctx, unsigned LID);
unsigned get_nout_CUDA(struct pr_CUDA_Context *ctx, unsigned LID);
void set_nout_plus_CUDA(struct pr_CUDA_Context *ctx, unsigned LID, unsigned nout);

