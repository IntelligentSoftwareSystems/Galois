#pragma once
#include "Galois/Runtime/Cuda/cuda_mtypes.h"
#include "Galois/Runtime/DataCommMode.h"

struct CUDA_Context;

struct CUDA_Context *get_CUDA_context(int id);
bool init_CUDA_context(struct CUDA_Context *ctx, int device);
void load_graph_CUDA(struct CUDA_Context *ctx, MarshalGraph &g, unsigned num_hosts);

void reset_CUDA_context(struct CUDA_Context *ctx);

void bitset_current_degree_clear_cuda(struct CUDA_Context *ctx);
unsigned int get_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void add_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void min_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void batch_get_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_slave_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_slave_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i);
void batch_get_reset_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode, unsigned int i);
void batch_set_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);

void bitset_flag_clear_cuda(struct CUDA_Context *ctx);
bool get_node_flag_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_flag_cuda(struct CUDA_Context *ctx, unsigned LID, bool v);
void add_node_flag_cuda(struct CUDA_Context *ctx, unsigned LID, bool v);
void min_node_flag_cuda(struct CUDA_Context *ctx, unsigned LID, bool v);
void batch_get_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, bool *v);
void batch_get_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_slave_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, bool *v);
void batch_get_slave_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, bool *v, bool i);
void batch_get_reset_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t *v_size, DataCommMode *data_mode, bool i);
void batch_set_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t v_size, DataCommMode data_mode);

void bitset_trim_clear_cuda(struct CUDA_Context *ctx);
unsigned int get_node_trim_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_trim_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void add_node_trim_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void min_node_trim_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void batch_get_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_slave_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_slave_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i);
void batch_get_reset_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode, unsigned int i);
void batch_set_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);

void InitializeGraph1_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void InitializeGraph1_all_cuda(struct CUDA_Context *ctx);
void InitializeGraph2_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void InitializeGraph2_all_cuda(struct CUDA_Context *ctx);
void KCoreStep1_cuda(unsigned int __begin, unsigned int __end, int & __retval, unsigned int local_k_core_num, struct CUDA_Context *ctx);
void KCoreStep1_all_cuda(int & __retval, unsigned int local_k_core_num, struct CUDA_Context *ctx);
void KCoreStep2_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void KCoreStep2_all_cuda(struct CUDA_Context *ctx);
