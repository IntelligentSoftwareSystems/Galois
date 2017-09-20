#pragma once
#include "galois/Runtime/Cuda/cuda_mtypes.h"
#include "galois/Runtime/DataCommMode.h"

struct CUDA_Context;

struct CUDA_Worklist {
	int *in_items;
	int num_in_items;
	int *out_items;
	int num_out_items;
	int max_size;
};

struct CUDA_Context *get_CUDA_context(int id);
bool init_CUDA_context(struct CUDA_Context *ctx, int device);
void load_graph_CUDA(struct CUDA_Context *ctx, struct CUDA_Worklist *wl, double wl_dup_factor, MarshalGraph &g, unsigned num_hosts);

void reset_CUDA_context(struct CUDA_Context *ctx);

void bitset_dist_current_clear_cuda(struct CUDA_Context *ctx);
unsigned int get_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void add_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void min_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void batch_get_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_mirror_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i);
void batch_get_reset_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode, unsigned int i);
void batch_set_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_dist_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);

void BFS_cuda(struct CUDA_Context *ctx);
void InitializeGraph_cuda(unsigned int __begin, unsigned int __end, const unsigned int & local_infinity, unsigned int local_src_node, struct CUDA_Context *ctx);
void InitializeGraph_all_cuda(const unsigned int & local_infinity, unsigned int local_src_node, struct CUDA_Context *ctx);
