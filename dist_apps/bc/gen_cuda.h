#pragma once
#include "galois/runtime/Cuda/cuda_mtypes.h"
#include "galois/runtime/DataCommMode.h"

struct CUDA_Context;

struct CUDA_Context *get_CUDA_context(int id);
bool init_CUDA_context(struct CUDA_Context *ctx, int device);
void load_graph_CUDA(struct CUDA_Context *ctx, MarshalGraph &g, unsigned num_hosts);

void reset_CUDA_context(struct CUDA_Context *ctx);

void bitset_betweeness_centrality_clear_cuda(struct CUDA_Context *ctx);
float get_node_betweeness_centrality_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_betweeness_centrality_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void add_node_betweeness_centrality_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void min_node_betweeness_centrality_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void batch_get_node_betweeness_centrality_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_get_node_betweeness_centrality_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_betweeness_centrality_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_get_mirror_node_betweeness_centrality_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_betweeness_centrality_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v, float i);
void batch_get_reset_node_betweeness_centrality_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode, float i);
void batch_set_node_betweeness_centrality_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_set_mirror_node_betweeness_centrality_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_betweeness_centrality_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_betweeness_centrality_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);

void bitset_current_length_clear_cuda(struct CUDA_Context *ctx);
uint32_t get_node_current_length_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_current_length_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void add_node_current_length_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
uint32_t min_node_current_length_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void batch_get_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v);
void batch_get_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v);
void batch_get_mirror_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v, uint32_t i);
void batch_get_reset_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode, uint32_t i);
void batch_set_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_set_mirror_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);

void bitset_dependency_clear_cuda(struct CUDA_Context *ctx);
float get_node_dependency_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_dependency_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void add_node_dependency_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void min_node_dependency_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void batch_get_node_dependency_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_get_node_dependency_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_dependency_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_get_mirror_node_dependency_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_dependency_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v, float i);
void batch_get_reset_node_dependency_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode, float i);
void batch_set_node_dependency_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_set_mirror_node_dependency_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_dependency_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_dependency_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);

void bitset_num_predecessors_clear_cuda(struct CUDA_Context *ctx);
uint32_t get_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void add_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void min_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void batch_get_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v);
void batch_get_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v);
void batch_get_mirror_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v, uint32_t i);
void batch_get_reset_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode, uint32_t i);
void batch_set_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_set_mirror_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);

void bitset_num_shortest_paths_clear_cuda(struct CUDA_Context *ctx);
uint32_t get_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void add_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void min_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void batch_get_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v);
void batch_get_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v);
void batch_get_mirror_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v, uint32_t i);
void batch_get_reset_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode, uint32_t i);
void batch_set_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_set_mirror_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);

void bitset_num_successors_clear_cuda(struct CUDA_Context *ctx);
uint32_t get_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void add_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void min_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void batch_get_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v);
void batch_get_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v);
void batch_get_mirror_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v, uint32_t i);
void batch_get_reset_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode, uint32_t i);
void batch_set_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_set_mirror_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);

void bitset_old_length_clear_cuda(struct CUDA_Context *ctx);
uint32_t get_node_old_length_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_old_length_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void add_node_old_length_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void min_node_old_length_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void batch_get_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v);
void batch_get_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v);
void batch_get_mirror_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v, uint32_t i);
void batch_get_reset_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode, uint32_t i);
void batch_set_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_set_mirror_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);

void bitset_propogation_flag_clear_cuda(struct CUDA_Context *ctx);
uint8_t get_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned LID, uint8_t v);
void add_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned LID, uint8_t v);
void min_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned LID, uint8_t v);
void batch_get_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v);
void batch_get_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint8_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v);
void batch_get_mirror_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint8_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, uint8_t i);
void batch_get_reset_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint8_t *v, size_t *v_size, DataCommMode *data_mode, uint8_t i);
void batch_set_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint8_t *v, size_t v_size, DataCommMode data_mode);
void batch_set_mirror_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint8_t *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint8_t *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint8_t *v, size_t v_size, DataCommMode data_mode);

void bitset_to_add_clear_cuda(struct CUDA_Context *ctx);
uint32_t get_node_to_add_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_to_add_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void add_node_to_add_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void min_node_to_add_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void batch_get_node_to_add_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v);
void batch_get_node_to_add_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_to_add_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v);
void batch_get_mirror_node_to_add_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_to_add_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v, uint32_t i);
void batch_get_reset_node_to_add_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode, uint32_t i);
void batch_set_node_to_add_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_set_mirror_node_to_add_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_to_add_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_to_add_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);

void bitset_to_add_float_clear_cuda(struct CUDA_Context *ctx);
float get_node_to_add_float_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_to_add_float_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void add_node_to_add_float_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void min_node_to_add_float_cuda(struct CUDA_Context *ctx, unsigned LID, float v);
void batch_get_node_to_add_float_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_get_node_to_add_float_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_to_add_float_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v);
void batch_get_mirror_node_to_add_float_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_to_add_float_cuda(struct CUDA_Context *ctx, unsigned from_id, float *v, float i);
void batch_get_reset_node_to_add_float_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode, float i);
void batch_set_node_to_add_float_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_set_mirror_node_to_add_float_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_to_add_float_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_to_add_float_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);

void bitset_trim_clear_cuda(struct CUDA_Context *ctx);
uint32_t get_node_trim_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_trim_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void add_node_trim_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void min_node_trim_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v);
void batch_get_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v);
void batch_get_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v);
void batch_get_mirror_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, uint32_t *v, uint32_t i);
void batch_get_reset_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode, uint32_t i);
void batch_set_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_set_mirror_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode);

void BC_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void BC_all_cuda(struct CUDA_Context *ctx);
void DependencyPropChanges_cuda(unsigned int __begin, unsigned int __end, const uint32_t & local_infinity, struct CUDA_Context *ctx);
void DependencyPropChanges_all_cuda(const uint32_t & local_infinity, struct CUDA_Context *ctx);
void DependencyPropogation_cuda(unsigned int __begin, unsigned int __end, int & __retval, const uint32_t & local_infinity, const uint64_t & local_current_src_node, struct CUDA_Context *ctx);
void DependencyPropogation_all_cuda(int & __retval, const uint32_t & local_infinity, const uint64_t & local_current_src_node, struct CUDA_Context *ctx);
void FirstIterationSSSP_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void FirstIterationSSSP_all_cuda(struct CUDA_Context *ctx);
void InitializeGraph_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void InitializeGraph_all_cuda(struct CUDA_Context *ctx);
void InitializeIteration_cuda(unsigned int __begin, unsigned int __end, const uint32_t & local_infinity, const uint64_t & local_current_src_node, struct CUDA_Context *ctx);
void InitializeIteration_all_cuda(const uint32_t & local_infinity, const uint64_t & local_current_src_node, struct CUDA_Context *ctx);
void NumShortestPaths_cuda(unsigned int __begin, unsigned int __end, int & __retval, const uint32_t & local_infinity, struct CUDA_Context *ctx);
void NumShortestPathsChanges_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void NumShortestPathsChanges_all_cuda(struct CUDA_Context *ctx);
void NumShortestPaths_all_cuda(int & __retval, const uint32_t & local_infinity, struct CUDA_Context *ctx);
void PredAndSucc_cuda(unsigned int __begin, unsigned int __end, const uint32_t & local_infinity, struct CUDA_Context *ctx);
void PredAndSucc_all_cuda(const uint32_t & local_infinity, struct CUDA_Context *ctx);
void SSSP_cuda(unsigned int __begin, unsigned int __end, int & __retval, struct CUDA_Context *ctx);
void SSSP_all_cuda(int & __retval, struct CUDA_Context *ctx);

// Bitset calls
void get_bitset_to_add_cuda(struct CUDA_Context *ctx, 
                              unsigned long long int *bitset_compute);
void bitset_to_add_reset_cuda(struct CUDA_Context *ctx);
void bitset_to_add_reset_cuda(struct CUDA_Context *ctx, size_t begin, 
                                size_t end);

void get_bitset_to_add_float_cuda(struct CUDA_Context *ctx, 
                              unsigned long long int *bitset_compute);
void bitset_to_add_float_reset_cuda(struct CUDA_Context *ctx);
void bitset_to_add_float_reset_cuda(struct CUDA_Context *ctx, size_t begin, 
                                size_t end);
void get_bitset_num_shortest_paths_cuda(struct CUDA_Context *ctx, 
                              unsigned long long int *bitset_compute);
void bitset_num_shortest_paths_reset_cuda(struct CUDA_Context *ctx);
void bitset_num_shortest_paths_reset_cuda(struct CUDA_Context *ctx, size_t begin, 
                                size_t end);
void get_bitset_num_successors_cuda(struct CUDA_Context *ctx, 
                              unsigned long long int *bitset_compute);
void bitset_num_successors_reset_cuda(struct CUDA_Context *ctx);
void bitset_num_successors_reset_cuda(struct CUDA_Context *ctx, size_t begin, 
                                size_t end);

void get_bitset_num_predecessors_cuda(struct CUDA_Context *ctx, 
                              unsigned long long int *bitset_compute);
void bitset_num_predecessors_reset_cuda(struct CUDA_Context *ctx);
void bitset_num_predecessors_reset_cuda(struct CUDA_Context *ctx, size_t begin, 
                                size_t end);

void get_bitset_trim_cuda(struct CUDA_Context *ctx, 
                              unsigned long long int *bitset_compute);
void bitset_trim_reset_cuda(struct CUDA_Context *ctx);
void bitset_trim_reset_cuda(struct CUDA_Context *ctx, size_t begin, 
                                size_t end);

void get_bitset_current_length_cuda(struct CUDA_Context *ctx, 
                              unsigned long long int *bitset_compute);
void bitset_current_length_reset_cuda(struct CUDA_Context *ctx);
void bitset_current_length_reset_cuda(struct CUDA_Context *ctx, size_t begin, 
                                size_t end);

void get_bitset_old_length_cuda(struct CUDA_Context *ctx, 
                              unsigned long long int *bitset_compute);
void bitset_old_length_reset_cuda(struct CUDA_Context *ctx);
void bitset_old_length_reset_cuda(struct CUDA_Context *ctx, size_t begin, 
                                size_t end);

void get_bitset_propogation_flag_cuda(struct CUDA_Context *ctx, 
                              unsigned long long int *bitset_compute);
void bitset_propogation_flag_reset_cuda(struct CUDA_Context *ctx);
void bitset_propogation_flag_reset_cuda(struct CUDA_Context *ctx, size_t begin, 
                                size_t end);

void get_bitset_dependency_cuda(struct CUDA_Context *ctx, 
                              unsigned long long int *bitset_compute);
void bitset_dependency_reset_cuda(struct CUDA_Context *ctx);
void bitset_dependency_reset_cuda(struct CUDA_Context *ctx, size_t begin, 
                                size_t end);
