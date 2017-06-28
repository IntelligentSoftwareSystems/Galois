#pragma once
#include "Galois/Runtime/Cuda/cuda_mtypes.h"
#include "Galois/Runtime/DataCommMode.h"

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
void batch_add_node_betweeness_centrality_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_betweeness_centrality_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);

void bitset_current_length_clear_cuda(struct CUDA_Context *ctx);
unsigned int get_node_current_length_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_current_length_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void add_node_current_length_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
bool min_node_current_length_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void batch_get_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_mirror_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i);
void batch_get_reset_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode, unsigned int i);
void batch_set_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_current_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);

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
void batch_add_node_dependency_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_dependency_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode);

void bitset_num_predecessors_clear_cuda(struct CUDA_Context *ctx);
unsigned int get_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void add_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void min_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void batch_get_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_mirror_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i);
void batch_get_reset_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode, unsigned int i);
void batch_set_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_num_predecessors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);

void bitset_num_shortest_paths_clear_cuda(struct CUDA_Context *ctx);
unsigned int get_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void add_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void min_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void batch_get_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_mirror_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i);
void batch_get_reset_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode, unsigned int i);
void batch_set_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_num_shortest_paths_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);

void bitset_num_successors_clear_cuda(struct CUDA_Context *ctx);
unsigned int get_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void add_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void min_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void batch_get_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_mirror_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i);
void batch_get_reset_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode, unsigned int i);
void batch_set_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_num_successors_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);

void bitset_old_length_clear_cuda(struct CUDA_Context *ctx);
unsigned int get_node_old_length_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_old_length_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void add_node_old_length_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void min_node_old_length_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void batch_get_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_mirror_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i);
void batch_get_reset_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode, unsigned int i);
void batch_set_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_old_length_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);

void bitset_propogation_flag_clear_cuda(struct CUDA_Context *ctx);
bool get_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned LID, bool v);
void add_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned LID, bool v);
void min_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned LID, bool v);
void batch_get_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, bool *v);
void batch_get_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, bool *v);
void batch_get_mirror_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, bool *v, bool i);
void batch_get_reset_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t *v_size, DataCommMode *data_mode, bool i);
void batch_set_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_propogation_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t v_size, DataCommMode data_mode);

void bitset_trim_clear_cuda(struct CUDA_Context *ctx);
unsigned int get_node_trim_cuda(struct CUDA_Context *ctx, unsigned LID);
void set_node_trim_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void add_node_trim_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void min_node_trim_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v);
void batch_get_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_mirror_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v);
void batch_get_mirror_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode);
void batch_get_reset_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i);
void batch_get_reset_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode, unsigned int i);
void batch_set_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_add_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);
void batch_min_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode);

void BC_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void BC_all_cuda(struct CUDA_Context *ctx);
void DependencyPropogation_cuda(unsigned int __begin, unsigned int __end, int & __retval, const unsigned int & local_current_src_node, struct CUDA_Context *ctx);
void DependencyPropogation_all_cuda(int & __retval, const unsigned int & local_current_src_node, struct CUDA_Context *ctx);
void FirstIterationSSSP_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void FirstIterationSSSP_all_cuda(struct CUDA_Context *ctx);
void InitializeGraph_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void InitializeGraph_all_cuda(struct CUDA_Context *ctx);
void InitializeIteration_cuda(unsigned int __begin, unsigned int __end, const unsigned int & local_infinity, const unsigned int & local_current_src_node, struct CUDA_Context *ctx);
void InitializeIteration_all_cuda(const unsigned int & local_infinity, const unsigned int & local_current_src_node, struct CUDA_Context *ctx);
void NumShortestPaths_cuda(unsigned int __begin, unsigned int __end, int & __retval, struct CUDA_Context *ctx);
void NumShortestPaths_all_cuda(int & __retval, struct CUDA_Context *ctx);
void PredAndSucc_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void PredAndSucc_all_cuda(struct CUDA_Context *ctx);
void PredecessorDecrement_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void PredecessorDecrement_all_cuda(struct CUDA_Context *ctx);
void PropFlagReset_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void PropFlagReset_all_cuda(struct CUDA_Context *ctx);
void SSSP_cuda(unsigned int __begin, unsigned int __end, int & __retval, struct CUDA_Context *ctx);
void SSSP_all_cuda(int & __retval, struct CUDA_Context *ctx);
void SuccessorDecrement_cuda(unsigned int __begin, unsigned int __end, struct CUDA_Context *ctx);
void SuccessorDecrement_all_cuda(struct CUDA_Context *ctx);
