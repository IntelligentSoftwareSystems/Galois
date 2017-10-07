#pragma once
#include "galois/runtime/DataCommMode.h"

struct CUDA_Context;

struct CUDA_Context *get_CUDA_context(int id);
bool init_CUDA_context(struct CUDA_Context *ctx, int device);
void load_graph_CUDA(struct CUDA_Context *ctx, MarshalGraph &g, unsigned num_hosts);

void reset_CUDA_context(struct CUDA_Context *ctx);

void get_bitset_comp_current_cuda(struct CUDA_Context *ctx, unsigned long long int *bitset_compute);
void bitset_comp_current_reset_cuda(struct CUDA_Context *ctx);
void bitset_comp_current_reset_cuda(struct CUDA_Context *ctx, size_t begin, size_t end);
unsigned long long get_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned long long v);
void add_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned long long v);
bool min_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned long long v);
void batch_get_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long *v);
void batch_get_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned long long *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long *v);
void batch_get_mirror_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned long long *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long *v, unsigned long long i);
void batch_get_reset_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned long long *v, size_t *v_size, DataCommMode *data_mode, unsigned long long i);
void batch_set_mirror_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned long long *v, size_t v_size, DataCommMode data_mode);
void batch_set_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned long long *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned long long *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned long long *v, size_t v_size, DataCommMode data_mode);

void ConnectedComp_cuda(unsigned int __begin, unsigned int __end, int & __retval, struct CUDA_Context *ctx);
void ConnectedComp_all_cuda(int & __retval, struct CUDA_Context *ctx);
void InitializeGraph_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void InitializeGraph_all_cuda(struct CUDA_Context *ctx);
