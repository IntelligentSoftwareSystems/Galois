#pragma once
#include "cuda/cuda_mtypes.h"

struct pr_CUDA_Context;

/* infrastructure */
struct pr_CUDA_Context *get_CUDA_context(int id);
bool init_CUDA_context(struct pr_CUDA_Context *ctx, int device);
void load_graph_CUDA(struct pr_CUDA_Context *ctx, MarshalGraph &g);

/* application message handlers */
unsigned int get_PRNode_nout_CUDA(struct pr_CUDA_Context *ctx, unsigned LID);
void set_PRNode_nout_CUDA(struct pr_CUDA_Context *ctx, unsigned LID, unsigned int nout);
void set_PRNode_nout_plus_CUDA(struct pr_CUDA_Context *ctx, unsigned LID, unsigned int nout);
float get_PRNode_pr_CUDA(struct pr_CUDA_Context *ctx, unsigned LID);
void set_PRNode_pr_CUDA(struct pr_CUDA_Context *ctx, unsigned LID, float pr);

/* application operator handlers */
void pr_cuda(struct pr_CUDA_Context *ctx);
void output_cuda(struct pr_CUDA_Context *ctx);
void initialize_node_cuda(struct pr_CUDA_Context *ctx);
