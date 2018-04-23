#pragma once
#include <cuda.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include "bc_push_cuda.h"
#include "galois/runtime/cuda/DeviceSync.h"

struct CUDA_Context : public CUDA_Context_Common {
	struct CUDA_Context_Field<float> betweeness_centrality;
	struct CUDA_Context_Field<uint32_t> current_length;
	struct CUDA_Context_Field<float> dependency;
	struct CUDA_Context_Field<uint32_t> num_predecessors;
	struct CUDA_Context_Field<uint64_t> num_shortest_paths;
	struct CUDA_Context_Field<uint32_t> num_successors;
	struct CUDA_Context_Field<uint32_t> old_length;
	struct CUDA_Context_Field<uint8_t> propagation_flag;
	struct CUDA_Context_Field<uint64_t> to_add;
	struct CUDA_Context_Field<float> to_add_float;
	struct CUDA_Context_Field<uint32_t> trim;
};

struct CUDA_Context* get_CUDA_context(int id) {
	struct CUDA_Context* ctx;
	ctx = (struct CUDA_Context* ) calloc(1, sizeof(struct CUDA_Context));
	ctx->id = id;
	return ctx;
}

bool init_CUDA_context(struct CUDA_Context* ctx, int device) {
	return init_CUDA_context_common(ctx, device);
}

void load_graph_CUDA(struct CUDA_Context* ctx, MarshalGraph &g, unsigned num_hosts) {
	size_t mem_usage = mem_usage_CUDA_common(g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->betweeness_centrality, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->current_length, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->dependency, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->num_predecessors, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->num_shortest_paths, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->num_successors, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->old_length, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->propagation_flag, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->to_add, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->to_add_float, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->trim, g, num_hosts);
	printf("[%d] Host memory for communication context: %3u MB\n", ctx->id, mem_usage/1048756);
	load_graph_CUDA_common(ctx, g, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->betweeness_centrality, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->current_length, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->dependency, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->num_predecessors, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->num_shortest_paths, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->num_successors, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->old_length, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->propagation_flag, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->to_add, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->to_add_float, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->trim, num_hosts);
	reset_CUDA_context(ctx);
}

void reset_CUDA_context(struct CUDA_Context* ctx) {
	ctx->betweeness_centrality.data.zero_gpu();
	ctx->current_length.data.zero_gpu();
	ctx->dependency.data.zero_gpu();
	ctx->num_predecessors.data.zero_gpu();
	ctx->num_shortest_paths.data.zero_gpu();
	ctx->num_successors.data.zero_gpu();
	ctx->old_length.data.zero_gpu();
	ctx->propagation_flag.data.zero_gpu();
	ctx->to_add.data.zero_gpu();
	ctx->to_add_float.data.zero_gpu();
	ctx->trim.data.zero_gpu();
}

void get_bitset_betweeness_centrality_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->betweeness_centrality.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_betweeness_centrality_reset_cuda(struct CUDA_Context* ctx) {
	ctx->betweeness_centrality.is_updated.cpu_rd_ptr()->reset();
}

void bitset_betweeness_centrality_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->betweeness_centrality, begin, end);
}

float get_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned LID) {
	float *betweeness_centrality = ctx->betweeness_centrality.data.cpu_rd_ptr();
	return betweeness_centrality[LID];
}

void set_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned LID, float v) {
	float *betweeness_centrality = ctx->betweeness_centrality.data.cpu_wr_ptr();
	betweeness_centrality[LID] = v;
}

void add_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned LID, float v) {
	float *betweeness_centrality = ctx->betweeness_centrality.data.cpu_wr_ptr();
	betweeness_centrality[LID] += v;
}

bool min_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned LID, float v) {
	float *betweeness_centrality = ctx->betweeness_centrality.data.cpu_wr_ptr();
	if (betweeness_centrality[LID] > v){
		betweeness_centrality[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, float *v) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->betweeness_centrality, from_id, v);
}

void batch_get_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->betweeness_centrality, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_mirror_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, float *v) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->betweeness_centrality, from_id, v);
}

void batch_get_mirror_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->betweeness_centrality, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_reset_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, float *v, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->betweeness_centrality, from_id, v, i);
}

void batch_get_reset_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->betweeness_centrality, from_id, bitset_comm, offsets, v, v_size, data_mode, i);
}

void batch_set_mirror_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMirror, setOp>(ctx, &ctx->betweeness_centrality, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_set_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, setOp>(ctx, &ctx->betweeness_centrality, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_add_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, addOp>(ctx, &ctx->betweeness_centrality, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_min_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, minOp>(ctx, &ctx->betweeness_centrality, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_reset_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, float v) {
	reset_data_field<float>(&ctx->betweeness_centrality, begin, end, v);
}

void get_bitset_current_length_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->current_length.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_current_length_reset_cuda(struct CUDA_Context* ctx) {
	ctx->current_length.is_updated.cpu_rd_ptr()->reset();
}

void bitset_current_length_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->current_length, begin, end);
}

uint32_t get_node_current_length_cuda(struct CUDA_Context* ctx, unsigned LID) {
	uint32_t *current_length = ctx->current_length.data.cpu_rd_ptr();
	return current_length[LID];
}

void set_node_current_length_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *current_length = ctx->current_length.data.cpu_wr_ptr();
	current_length[LID] = v;
}

void add_node_current_length_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *current_length = ctx->current_length.data.cpu_wr_ptr();
	current_length[LID] += v;
}

bool min_node_current_length_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *current_length = ctx->current_length.data.cpu_wr_ptr();
	if (current_length[LID] > v){
		current_length[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->current_length, from_id, v);
}

void batch_get_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->current_length, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_mirror_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->current_length, from_id, v);
}

void batch_get_mirror_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->current_length, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_reset_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->current_length, from_id, v, i);
}

void batch_get_reset_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->current_length, from_id, bitset_comm, offsets, v, v_size, data_mode, i);
}

void batch_set_mirror_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, setOp>(ctx, &ctx->current_length, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_set_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, setOp>(ctx, &ctx->current_length, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_add_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, addOp>(ctx, &ctx->current_length, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_min_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, minOp>(ctx, &ctx->current_length, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_reset_node_current_length_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, uint32_t v) {
	reset_data_field<uint32_t>(&ctx->current_length, begin, end, v);
}

void get_bitset_dependency_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->dependency.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_dependency_reset_cuda(struct CUDA_Context* ctx) {
	ctx->dependency.is_updated.cpu_rd_ptr()->reset();
}

void bitset_dependency_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->dependency, begin, end);
}

float get_node_dependency_cuda(struct CUDA_Context* ctx, unsigned LID) {
	float *dependency = ctx->dependency.data.cpu_rd_ptr();
	return dependency[LID];
}

void set_node_dependency_cuda(struct CUDA_Context* ctx, unsigned LID, float v) {
	float *dependency = ctx->dependency.data.cpu_wr_ptr();
	dependency[LID] = v;
}

void add_node_dependency_cuda(struct CUDA_Context* ctx, unsigned LID, float v) {
	float *dependency = ctx->dependency.data.cpu_wr_ptr();
	dependency[LID] += v;
}

bool min_node_dependency_cuda(struct CUDA_Context* ctx, unsigned LID, float v) {
	float *dependency = ctx->dependency.data.cpu_wr_ptr();
	if (dependency[LID] > v){
		dependency[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, float *v) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->dependency, from_id, v);
}

void batch_get_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->dependency, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, float *v) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->dependency, from_id, v);
}

void batch_get_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->dependency, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_reset_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, float *v, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->dependency, from_id, v, i);
}

void batch_get_reset_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->dependency, from_id, bitset_comm, offsets, v, v_size, data_mode, i);
}

void batch_set_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMirror, setOp>(ctx, &ctx->dependency, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_set_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, setOp>(ctx, &ctx->dependency, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_add_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, addOp>(ctx, &ctx->dependency, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_min_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, minOp>(ctx, &ctx->dependency, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_reset_node_dependency_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, float v) {
	reset_data_field<float>(&ctx->dependency, begin, end, v);
}

void get_bitset_num_predecessors_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->num_predecessors.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_num_predecessors_reset_cuda(struct CUDA_Context* ctx) {
	ctx->num_predecessors.is_updated.cpu_rd_ptr()->reset();
}

void bitset_num_predecessors_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->num_predecessors, begin, end);
}

uint32_t get_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned LID) {
	uint32_t *num_predecessors = ctx->num_predecessors.data.cpu_rd_ptr();
	return num_predecessors[LID];
}

void set_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *num_predecessors = ctx->num_predecessors.data.cpu_wr_ptr();
	num_predecessors[LID] = v;
}

void add_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *num_predecessors = ctx->num_predecessors.data.cpu_wr_ptr();
	num_predecessors[LID] += v;
}

bool min_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *num_predecessors = ctx->num_predecessors.data.cpu_wr_ptr();
	if (num_predecessors[LID] > v){
		num_predecessors[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->num_predecessors, from_id, v);
}

void batch_get_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->num_predecessors, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_mirror_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->num_predecessors, from_id, v);
}

void batch_get_mirror_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->num_predecessors, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_reset_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->num_predecessors, from_id, v, i);
}

void batch_get_reset_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->num_predecessors, from_id, bitset_comm, offsets, v, v_size, data_mode, i);
}

void batch_set_mirror_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, setOp>(ctx, &ctx->num_predecessors, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_set_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, setOp>(ctx, &ctx->num_predecessors, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_add_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, addOp>(ctx, &ctx->num_predecessors, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_min_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, minOp>(ctx, &ctx->num_predecessors, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_reset_node_num_predecessors_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, uint32_t v) {
	reset_data_field<uint32_t>(&ctx->num_predecessors, begin, end, v);
}

void get_bitset_num_shortest_paths_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->num_shortest_paths.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_num_shortest_paths_reset_cuda(struct CUDA_Context* ctx) {
	ctx->num_shortest_paths.is_updated.cpu_rd_ptr()->reset();
}

void bitset_num_shortest_paths_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->num_shortest_paths, begin, end);
}

uint64_t get_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned LID) {
	uint64_t *num_shortest_paths = ctx->num_shortest_paths.data.cpu_rd_ptr();
	return num_shortest_paths[LID];
}

void set_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned LID, uint64_t v) {
	uint64_t *num_shortest_paths = ctx->num_shortest_paths.data.cpu_wr_ptr();
	num_shortest_paths[LID] = v;
}

void add_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned LID, uint64_t v) {
	uint64_t *num_shortest_paths = ctx->num_shortest_paths.data.cpu_wr_ptr();
	num_shortest_paths[LID] += v;
}

bool min_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned LID, uint64_t v) {
	uint64_t *num_shortest_paths = ctx->num_shortest_paths.data.cpu_wr_ptr();
	if (num_shortest_paths[LID] > v){
		num_shortest_paths[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *v) {
	batch_get_shared_field<uint64_t, sharedMaster, false>(ctx, &ctx->num_shortest_paths, from_id, v);
}

void batch_get_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint64_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint64_t, sharedMaster, false>(ctx, &ctx->num_shortest_paths, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_mirror_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *v) {
	batch_get_shared_field<uint64_t, sharedMirror, false>(ctx, &ctx->num_shortest_paths, from_id, v);
}

void batch_get_mirror_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint64_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint64_t, sharedMirror, false>(ctx, &ctx->num_shortest_paths, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_reset_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *v, uint64_t i) {
	batch_get_shared_field<uint64_t, sharedMirror, true>(ctx, &ctx->num_shortest_paths, from_id, v, i);
}

void batch_get_reset_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint64_t *v, size_t *v_size, DataCommMode *data_mode, uint64_t i) {
	batch_get_shared_field<uint64_t, sharedMirror, true>(ctx, &ctx->num_shortest_paths, from_id, bitset_comm, offsets, v, v_size, data_mode, i);
}

void batch_set_mirror_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint64_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint64_t, sharedMirror, setOp>(ctx, &ctx->num_shortest_paths, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_set_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint64_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint64_t, sharedMaster, setOp>(ctx, &ctx->num_shortest_paths, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_add_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint64_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint64_t, sharedMaster, addOp>(ctx, &ctx->num_shortest_paths, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_min_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint64_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint64_t, sharedMaster, minOp>(ctx, &ctx->num_shortest_paths, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_reset_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, uint64_t v) {
	reset_data_field<uint64_t>(&ctx->num_shortest_paths, begin, end, v);
}

void get_bitset_num_successors_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->num_successors.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_num_successors_reset_cuda(struct CUDA_Context* ctx) {
	ctx->num_successors.is_updated.cpu_rd_ptr()->reset();
}

void bitset_num_successors_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->num_successors, begin, end);
}

uint32_t get_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned LID) {
	uint32_t *num_successors = ctx->num_successors.data.cpu_rd_ptr();
	return num_successors[LID];
}

void set_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *num_successors = ctx->num_successors.data.cpu_wr_ptr();
	num_successors[LID] = v;
}

void add_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *num_successors = ctx->num_successors.data.cpu_wr_ptr();
	num_successors[LID] += v;
}

bool min_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *num_successors = ctx->num_successors.data.cpu_wr_ptr();
	if (num_successors[LID] > v){
		num_successors[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->num_successors, from_id, v);
}

void batch_get_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->num_successors, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_mirror_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->num_successors, from_id, v);
}

void batch_get_mirror_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->num_successors, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_reset_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->num_successors, from_id, v, i);
}

void batch_get_reset_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->num_successors, from_id, bitset_comm, offsets, v, v_size, data_mode, i);
}

void batch_set_mirror_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, setOp>(ctx, &ctx->num_successors, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_set_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, setOp>(ctx, &ctx->num_successors, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_add_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, addOp>(ctx, &ctx->num_successors, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_min_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, minOp>(ctx, &ctx->num_successors, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_reset_node_num_successors_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, uint32_t v) {
	reset_data_field<uint32_t>(&ctx->num_successors, begin, end, v);
}

void get_bitset_old_length_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->old_length.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_old_length_reset_cuda(struct CUDA_Context* ctx) {
	ctx->old_length.is_updated.cpu_rd_ptr()->reset();
}

void bitset_old_length_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->old_length, begin, end);
}

uint32_t get_node_old_length_cuda(struct CUDA_Context* ctx, unsigned LID) {
	uint32_t *old_length = ctx->old_length.data.cpu_rd_ptr();
	return old_length[LID];
}

void set_node_old_length_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *old_length = ctx->old_length.data.cpu_wr_ptr();
	old_length[LID] = v;
}

void add_node_old_length_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *old_length = ctx->old_length.data.cpu_wr_ptr();
	old_length[LID] += v;
}

bool min_node_old_length_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *old_length = ctx->old_length.data.cpu_wr_ptr();
	if (old_length[LID] > v){
		old_length[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_old_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->old_length, from_id, v);
}

void batch_get_node_old_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->old_length, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_mirror_node_old_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->old_length, from_id, v);
}

void batch_get_mirror_node_old_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->old_length, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_reset_node_old_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->old_length, from_id, v, i);
}

void batch_get_reset_node_old_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->old_length, from_id, bitset_comm, offsets, v, v_size, data_mode, i);
}

void batch_set_mirror_node_old_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, setOp>(ctx, &ctx->old_length, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_set_node_old_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, setOp>(ctx, &ctx->old_length, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_add_node_old_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, addOp>(ctx, &ctx->old_length, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_min_node_old_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, minOp>(ctx, &ctx->old_length, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_reset_node_old_length_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, uint32_t v) {
	reset_data_field<uint32_t>(&ctx->old_length, begin, end, v);
}

void get_bitset_propagation_flag_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->propagation_flag.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_propagation_flag_reset_cuda(struct CUDA_Context* ctx) {
	ctx->propagation_flag.is_updated.cpu_rd_ptr()->reset();
}

void bitset_propagation_flag_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->propagation_flag, begin, end);
}

uint8_t get_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned LID) {
	uint8_t *propagation_flag = ctx->propagation_flag.data.cpu_rd_ptr();
	return propagation_flag[LID];
}

void set_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned LID, uint8_t v) {
	uint8_t *propagation_flag = ctx->propagation_flag.data.cpu_wr_ptr();
	propagation_flag[LID] = v;
}

void add_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned LID, uint8_t v) {
	uint8_t *propagation_flag = ctx->propagation_flag.data.cpu_wr_ptr();
	propagation_flag[LID] += v;
}

bool min_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned LID, uint8_t v) {
	uint8_t *propagation_flag = ctx->propagation_flag.data.cpu_wr_ptr();
	if (propagation_flag[LID] > v){
		propagation_flag[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v) {
	batch_get_shared_field<uint8_t, sharedMaster, false>(ctx, &ctx->propagation_flag, from_id, v);
}

void batch_get_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint8_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint8_t, sharedMaster, false>(ctx, &ctx->propagation_flag, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_mirror_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v) {
	batch_get_shared_field<uint8_t, sharedMirror, false>(ctx, &ctx->propagation_flag, from_id, v);
}

void batch_get_mirror_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint8_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint8_t, sharedMirror, false>(ctx, &ctx->propagation_flag, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_reset_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, uint8_t i) {
	batch_get_shared_field<uint8_t, sharedMirror, true>(ctx, &ctx->propagation_flag, from_id, v, i);
}

void batch_get_reset_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint8_t *v, size_t *v_size, DataCommMode *data_mode, uint8_t i) {
	batch_get_shared_field<uint8_t, sharedMirror, true>(ctx, &ctx->propagation_flag, from_id, bitset_comm, offsets, v, v_size, data_mode, i);
}

void batch_set_mirror_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint8_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint8_t, sharedMirror, setOp>(ctx, &ctx->propagation_flag, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_set_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint8_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint8_t, sharedMaster, setOp>(ctx, &ctx->propagation_flag, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_add_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint8_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint8_t, sharedMaster, addOp>(ctx, &ctx->propagation_flag, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_min_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint8_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint8_t, sharedMaster, minOp>(ctx, &ctx->propagation_flag, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_reset_node_propagation_flag_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, uint8_t v) {
	reset_data_field<uint8_t>(&ctx->propagation_flag, begin, end, v);
}

void get_bitset_to_add_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->to_add.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_to_add_reset_cuda(struct CUDA_Context* ctx) {
	ctx->to_add.is_updated.cpu_rd_ptr()->reset();
}

void bitset_to_add_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->to_add, begin, end);
}

uint64_t get_node_to_add_cuda(struct CUDA_Context* ctx, unsigned LID) {
	uint64_t *to_add = ctx->to_add.data.cpu_rd_ptr();
	return to_add[LID];
}

void set_node_to_add_cuda(struct CUDA_Context* ctx, unsigned LID, uint64_t v) {
	uint64_t *to_add = ctx->to_add.data.cpu_wr_ptr();
	to_add[LID] = v;
}

void add_node_to_add_cuda(struct CUDA_Context* ctx, unsigned LID, uint64_t v) {
	uint64_t *to_add = ctx->to_add.data.cpu_wr_ptr();
	to_add[LID] += v;
}

bool min_node_to_add_cuda(struct CUDA_Context* ctx, unsigned LID, uint64_t v) {
	uint64_t *to_add = ctx->to_add.data.cpu_wr_ptr();
	if (to_add[LID] > v){
		to_add[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_to_add_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *v) {
	batch_get_shared_field<uint64_t, sharedMaster, false>(ctx, &ctx->to_add, from_id, v);
}

void batch_get_node_to_add_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint64_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint64_t, sharedMaster, false>(ctx, &ctx->to_add, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_mirror_node_to_add_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *v) {
	batch_get_shared_field<uint64_t, sharedMirror, false>(ctx, &ctx->to_add, from_id, v);
}

void batch_get_mirror_node_to_add_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint64_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint64_t, sharedMirror, false>(ctx, &ctx->to_add, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_reset_node_to_add_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *v, uint64_t i) {
	batch_get_shared_field<uint64_t, sharedMirror, true>(ctx, &ctx->to_add, from_id, v, i);
}

void batch_get_reset_node_to_add_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint64_t *v, size_t *v_size, DataCommMode *data_mode, uint64_t i) {
	batch_get_shared_field<uint64_t, sharedMirror, true>(ctx, &ctx->to_add, from_id, bitset_comm, offsets, v, v_size, data_mode, i);
}

void batch_set_mirror_node_to_add_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint64_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint64_t, sharedMirror, setOp>(ctx, &ctx->to_add, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_set_node_to_add_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint64_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint64_t, sharedMaster, setOp>(ctx, &ctx->to_add, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_add_node_to_add_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint64_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint64_t, sharedMaster, addOp>(ctx, &ctx->to_add, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_min_node_to_add_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint64_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint64_t, sharedMaster, minOp>(ctx, &ctx->to_add, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_reset_node_to_add_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, uint64_t v) {
	reset_data_field<uint64_t>(&ctx->to_add, begin, end, v);
}

void get_bitset_to_add_float_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->to_add_float.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_to_add_float_reset_cuda(struct CUDA_Context* ctx) {
	ctx->to_add_float.is_updated.cpu_rd_ptr()->reset();
}

void bitset_to_add_float_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->to_add_float, begin, end);
}

float get_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned LID) {
	float *to_add_float = ctx->to_add_float.data.cpu_rd_ptr();
	return to_add_float[LID];
}

void set_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned LID, float v) {
	float *to_add_float = ctx->to_add_float.data.cpu_wr_ptr();
	to_add_float[LID] = v;
}

void add_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned LID, float v) {
	float *to_add_float = ctx->to_add_float.data.cpu_wr_ptr();
	to_add_float[LID] += v;
}

bool min_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned LID, float v) {
	float *to_add_float = ctx->to_add_float.data.cpu_wr_ptr();
	if (to_add_float[LID] > v){
		to_add_float[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned from_id, float *v) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->to_add_float, from_id, v);
}

void batch_get_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->to_add_float, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_mirror_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned from_id, float *v) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->to_add_float, from_id, v);
}

void batch_get_mirror_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->to_add_float, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_reset_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned from_id, float *v, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->to_add_float, from_id, v, i);
}

void batch_get_reset_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t *v_size, DataCommMode *data_mode, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->to_add_float, from_id, bitset_comm, offsets, v, v_size, data_mode, i);
}

void batch_set_mirror_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMirror, setOp>(ctx, &ctx->to_add_float, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_set_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, setOp>(ctx, &ctx->to_add_float, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_add_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, addOp>(ctx, &ctx->to_add_float, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_min_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, float *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, minOp>(ctx, &ctx->to_add_float, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_reset_node_to_add_float_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, float v) {
	reset_data_field<float>(&ctx->to_add_float, begin, end, v);
}

void get_bitset_trim_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->trim.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_trim_reset_cuda(struct CUDA_Context* ctx) {
	ctx->trim.is_updated.cpu_rd_ptr()->reset();
}

void bitset_trim_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->trim, begin, end);
}

uint32_t get_node_trim_cuda(struct CUDA_Context* ctx, unsigned LID) {
	uint32_t *trim = ctx->trim.data.cpu_rd_ptr();
	return trim[LID];
}

void set_node_trim_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *trim = ctx->trim.data.cpu_wr_ptr();
	trim[LID] = v;
}

void add_node_trim_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *trim = ctx->trim.data.cpu_wr_ptr();
	trim[LID] += v;
}

bool min_node_trim_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *trim = ctx->trim.data.cpu_wr_ptr();
	if (trim[LID] > v){
		trim[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->trim, from_id, v);
}

void batch_get_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->trim, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_mirror_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->trim, from_id, v);
}

void batch_get_mirror_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->trim, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_reset_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint32_t *v, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->trim, from_id, v, i);
}

void batch_get_reset_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t *v_size, DataCommMode *data_mode, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->trim, from_id, bitset_comm, offsets, v, v_size, data_mode, i);
}

void batch_set_mirror_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, setOp>(ctx, &ctx->trim, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_set_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, setOp>(ctx, &ctx->trim, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_add_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, addOp>(ctx, &ctx->trim, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_min_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint64_t *bitset_comm, unsigned int *offsets, uint32_t *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, minOp>(ctx, &ctx->trim, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_reset_node_trim_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, uint32_t v) {
	reset_data_field<uint32_t>(&ctx->trim, begin, end, v);
}

