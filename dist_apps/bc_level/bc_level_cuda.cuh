#pragma once
#include <cuda.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include "bc_level_cuda.h"
#include "galois/runtime/cuda/DeviceSync.h"

struct CUDA_Context : public CUDA_Context_Common {
	struct CUDA_Context_Field<float> betweeness_centrality;
	struct CUDA_Context_Field<uint32_t> current_length;
	struct CUDA_Context_Field<float> dependency;
	struct CUDA_Context_Field<ShortPathType> num_shortest_paths;
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
	mem_usage += mem_usage_CUDA_field(&ctx->num_shortest_paths, g, num_hosts);
	printf("[%d] Host memory for communication context: %3u MB\n", ctx->id, mem_usage/1048756);
	load_graph_CUDA_common(ctx, g, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->betweeness_centrality, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->current_length, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->dependency, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->num_shortest_paths, num_hosts);
	reset_CUDA_context(ctx);
}

void reset_CUDA_context(struct CUDA_Context* ctx) {
	ctx->betweeness_centrality.data.zero_gpu();
	ctx->current_length.data.zero_gpu();
	ctx->dependency.data.zero_gpu();
	ctx->num_shortest_paths.data.zero_gpu();
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

void batch_get_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->betweeness_centrality, from_id, v);
}

void batch_get_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->betweeness_centrality, from_id, v, v_size, data_mode);
}

void batch_get_mirror_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->betweeness_centrality, from_id, v);
}

void batch_get_mirror_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->betweeness_centrality, from_id, v, v_size, data_mode);
}

void batch_get_reset_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->betweeness_centrality, from_id, v, i);
}

void batch_get_reset_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->betweeness_centrality, from_id, v, v_size, data_mode, i);
}

void batch_set_mirror_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMirror, setOp>(ctx, &ctx->betweeness_centrality, from_id, v, data_mode);
}

void batch_set_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, setOp>(ctx, &ctx->betweeness_centrality, from_id, v, data_mode);
}

void batch_add_mirror_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMirror, addOp>(ctx, &ctx->betweeness_centrality, from_id, v, data_mode);
}

void batch_add_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, addOp>(ctx, &ctx->betweeness_centrality, from_id, v, data_mode);
}

void batch_min_mirror_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMirror, minOp>(ctx, &ctx->betweeness_centrality, from_id, v, data_mode);
}

void batch_min_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, minOp>(ctx, &ctx->betweeness_centrality, from_id, v, data_mode);
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

void batch_get_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->current_length, from_id, v);
}

void batch_get_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->current_length, from_id, v, v_size, data_mode);
}

void batch_get_mirror_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->current_length, from_id, v);
}

void batch_get_mirror_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->current_length, from_id, v, v_size, data_mode);
}

void batch_get_reset_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->current_length, from_id, v, i);
}

void batch_get_reset_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->current_length, from_id, v, v_size, data_mode, i);
}

void batch_set_mirror_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, setOp>(ctx, &ctx->current_length, from_id, v, data_mode);
}

void batch_set_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, setOp>(ctx, &ctx->current_length, from_id, v, data_mode);
}

void batch_add_mirror_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, addOp>(ctx, &ctx->current_length, from_id, v, data_mode);
}

void batch_add_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, addOp>(ctx, &ctx->current_length, from_id, v, data_mode);
}

void batch_min_mirror_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, minOp>(ctx, &ctx->current_length, from_id, v, data_mode);
}

void batch_min_node_current_length_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, minOp>(ctx, &ctx->current_length, from_id, v, data_mode);
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

void batch_get_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->dependency, from_id, v);
}

void batch_get_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->dependency, from_id, v, v_size, data_mode);
}

void batch_get_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->dependency, from_id, v);
}

void batch_get_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->dependency, from_id, v, v_size, data_mode);
}

void batch_get_reset_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->dependency, from_id, v, i);
}

void batch_get_reset_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->dependency, from_id, v, v_size, data_mode, i);
}

void batch_set_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMirror, setOp>(ctx, &ctx->dependency, from_id, v, data_mode);
}

void batch_set_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, setOp>(ctx, &ctx->dependency, from_id, v, data_mode);
}

void batch_add_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMirror, addOp>(ctx, &ctx->dependency, from_id, v, data_mode);
}

void batch_add_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, addOp>(ctx, &ctx->dependency, from_id, v, data_mode);
}

void batch_min_mirror_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMirror, minOp>(ctx, &ctx->dependency, from_id, v, data_mode);
}

void batch_min_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, minOp>(ctx, &ctx->dependency, from_id, v, data_mode);
}

void batch_reset_node_dependency_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, float v) {
	reset_data_field<float>(&ctx->dependency, begin, end, v);
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

ShortPathType get_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned LID) {
	ShortPathType *num_shortest_paths = ctx->num_shortest_paths.data.cpu_rd_ptr();
	return num_shortest_paths[LID];
}

void set_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned LID, ShortPathType v) {
	ShortPathType *num_shortest_paths = ctx->num_shortest_paths.data.cpu_wr_ptr();
	num_shortest_paths[LID] = v;
}

void add_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned LID, ShortPathType v) {
	ShortPathType *num_shortest_paths = ctx->num_shortest_paths.data.cpu_wr_ptr();
	num_shortest_paths[LID] += v;
}

bool min_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned LID, ShortPathType v) {
	ShortPathType *num_shortest_paths = ctx->num_shortest_paths.data.cpu_wr_ptr();
	if (num_shortest_paths[LID] > v){
		num_shortest_paths[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<ShortPathType, sharedMaster, false>(ctx, &ctx->num_shortest_paths, from_id, v);
}

void batch_get_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<ShortPathType, sharedMaster, false>(ctx, &ctx->num_shortest_paths, from_id, v, v_size, data_mode);
}

void batch_get_mirror_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<ShortPathType, sharedMirror, false>(ctx, &ctx->num_shortest_paths, from_id, v);
}

void batch_get_mirror_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<ShortPathType, sharedMirror, false>(ctx, &ctx->num_shortest_paths, from_id, v, v_size, data_mode);
}

void batch_get_reset_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, ShortPathType i) {
	batch_get_shared_field<ShortPathType, sharedMirror, true>(ctx, &ctx->num_shortest_paths, from_id, v, i);
}

void batch_get_reset_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, ShortPathType i) {
	batch_get_shared_field<ShortPathType, sharedMirror, true>(ctx, &ctx->num_shortest_paths, from_id, v, v_size, data_mode, i);
}

void batch_set_mirror_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<ShortPathType, sharedMirror, setOp>(ctx, &ctx->num_shortest_paths, from_id, v, data_mode);
}

void batch_set_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<ShortPathType, sharedMaster, setOp>(ctx, &ctx->num_shortest_paths, from_id, v, data_mode);
}

void batch_add_mirror_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<ShortPathType, sharedMirror, addOp>(ctx, &ctx->num_shortest_paths, from_id, v, data_mode);
}

void batch_add_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<ShortPathType, sharedMaster, addOp>(ctx, &ctx->num_shortest_paths, from_id, v, data_mode);
}

void batch_min_mirror_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<ShortPathType, sharedMirror, minOp>(ctx, &ctx->num_shortest_paths, from_id, v, data_mode);
}

void batch_min_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<ShortPathType, sharedMaster, minOp>(ctx, &ctx->num_shortest_paths, from_id, v, data_mode);
}

void batch_reset_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, ShortPathType v) {
	reset_data_field<ShortPathType>(&ctx->num_shortest_paths, begin, end, v);
}

