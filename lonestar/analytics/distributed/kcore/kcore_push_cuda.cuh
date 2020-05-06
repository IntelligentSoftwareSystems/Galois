#pragma once
#include <cuda.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include "kcore_push_cuda.h"
#include "galois/runtime/cuda/DeviceSync.h"

struct CUDA_Context : public CUDA_Context_Common {
	struct CUDA_Context_Field<uint32_t> current_degree;
	struct CUDA_Context_Field<uint8_t> flag;
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
	mem_usage += mem_usage_CUDA_field(&ctx->current_degree, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->flag, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->trim, g, num_hosts);
	printf("[%d] Host memory for communication context: %3u MB\n", ctx->id, mem_usage/1048756);
	load_graph_CUDA_common(ctx, g, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->current_degree, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->flag, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->trim, num_hosts);
	reset_CUDA_context(ctx);
}

void reset_CUDA_context(struct CUDA_Context* ctx) {
	ctx->current_degree.data.zero_gpu();
	ctx->flag.data.zero_gpu();
	ctx->trim.data.zero_gpu();
}

void get_bitset_current_degree_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->current_degree.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_current_degree_reset_cuda(struct CUDA_Context* ctx) {
	ctx->current_degree.is_updated.cpu_rd_ptr()->reset();
}

void bitset_current_degree_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->current_degree, begin, end);
}

uint32_t get_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned LID) {
	uint32_t *current_degree = ctx->current_degree.data.cpu_rd_ptr();
	return current_degree[LID];
}

void set_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *current_degree = ctx->current_degree.data.cpu_wr_ptr();
	current_degree[LID] = v;
}

void add_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *current_degree = ctx->current_degree.data.cpu_wr_ptr();
	current_degree[LID] += v;
}

bool min_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *current_degree = ctx->current_degree.data.cpu_wr_ptr();
	if (current_degree[LID] > v){
		current_degree[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->current_degree, from_id, v);
}

void batch_get_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->current_degree, from_id, v, v_size, data_mode);
}

void batch_get_mirror_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->current_degree, from_id, v);
}

void batch_get_mirror_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->current_degree, from_id, v, v_size, data_mode);
}

void batch_get_reset_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->current_degree, from_id, v, i);
}

void batch_get_reset_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->current_degree, from_id, v, v_size, data_mode, i);
}

void batch_set_mirror_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, setOp>(ctx, &ctx->current_degree, from_id, v, data_mode);
}

void batch_set_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, setOp>(ctx, &ctx->current_degree, from_id, v, data_mode);
}

void batch_add_mirror_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, addOp>(ctx, &ctx->current_degree, from_id, v, data_mode);
}

void batch_add_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, addOp>(ctx, &ctx->current_degree, from_id, v, data_mode);
}

void batch_min_mirror_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, minOp>(ctx, &ctx->current_degree, from_id, v, data_mode);
}

void batch_min_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, minOp>(ctx, &ctx->current_degree, from_id, v, data_mode);
}

void batch_reset_node_current_degree_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, uint32_t v) {
	reset_data_field<uint32_t>(&ctx->current_degree, begin, end, v);
}

void get_bitset_flag_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->flag.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_flag_reset_cuda(struct CUDA_Context* ctx) {
	ctx->flag.is_updated.cpu_rd_ptr()->reset();
}

void bitset_flag_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->flag, begin, end);
}

uint8_t get_node_flag_cuda(struct CUDA_Context* ctx, unsigned LID) {
	uint8_t *flag = ctx->flag.data.cpu_rd_ptr();
	return flag[LID];
}

void set_node_flag_cuda(struct CUDA_Context* ctx, unsigned LID, uint8_t v) {
	uint8_t *flag = ctx->flag.data.cpu_wr_ptr();
	flag[LID] = v;
}

void add_node_flag_cuda(struct CUDA_Context* ctx, unsigned LID, uint8_t v) {
	uint8_t *flag = ctx->flag.data.cpu_wr_ptr();
	flag[LID] += v;
}

bool min_node_flag_cuda(struct CUDA_Context* ctx, unsigned LID, uint8_t v) {
	uint8_t *flag = ctx->flag.data.cpu_wr_ptr();
	if (flag[LID] > v){
		flag[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<uint8_t, sharedMaster, false>(ctx, &ctx->flag, from_id, v);
}

void batch_get_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<uint8_t, sharedMaster, false>(ctx, &ctx->flag, from_id, v, v_size, data_mode);
}

void batch_get_mirror_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<uint8_t, sharedMirror, false>(ctx, &ctx->flag, from_id, v);
}

void batch_get_mirror_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<uint8_t, sharedMirror, false>(ctx, &ctx->flag, from_id, v, v_size, data_mode);
}

void batch_get_reset_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, uint8_t i) {
	batch_get_shared_field<uint8_t, sharedMirror, true>(ctx, &ctx->flag, from_id, v, i);
}

void batch_get_reset_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, uint8_t i) {
	batch_get_shared_field<uint8_t, sharedMirror, true>(ctx, &ctx->flag, from_id, v, v_size, data_mode, i);
}

void batch_set_mirror_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint8_t, sharedMirror, setOp>(ctx, &ctx->flag, from_id, v, data_mode);
}

void batch_set_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint8_t, sharedMaster, setOp>(ctx, &ctx->flag, from_id, v, data_mode);
}

void batch_add_mirror_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint8_t, sharedMirror, addOp>(ctx, &ctx->flag, from_id, v, data_mode);
}

void batch_add_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint8_t, sharedMaster, addOp>(ctx, &ctx->flag, from_id, v, data_mode);
}

void batch_min_mirror_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint8_t, sharedMirror, minOp>(ctx, &ctx->flag, from_id, v, data_mode);
}

void batch_min_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint8_t, sharedMaster, minOp>(ctx, &ctx->flag, from_id, v, data_mode);
}

void batch_reset_node_flag_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, uint8_t v) {
	reset_data_field<uint8_t>(&ctx->flag, begin, end, v);
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

void batch_get_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->trim, from_id, v);
}

void batch_get_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->trim, from_id, v, v_size, data_mode);
}

void batch_get_mirror_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->trim, from_id, v);
}

void batch_get_mirror_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->trim, from_id, v, v_size, data_mode);
}

void batch_get_reset_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->trim, from_id, v, i);
}

void batch_get_reset_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, size_t* v_size, DataCommMode* data_mode, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->trim, from_id, v, v_size, data_mode, i);
}

void batch_set_mirror_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, setOp>(ctx, &ctx->trim, from_id, v, data_mode);
}

void batch_set_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, setOp>(ctx, &ctx->trim, from_id, v, data_mode);
}

void batch_add_mirror_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, addOp>(ctx, &ctx->trim, from_id, v, data_mode);
}

void batch_add_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, addOp>(ctx, &ctx->trim, from_id, v, data_mode);
}

void batch_min_mirror_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, minOp>(ctx, &ctx->trim, from_id, v, data_mode);
}

void batch_min_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t* v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, minOp>(ctx, &ctx->trim, from_id, v, data_mode);
}

void batch_reset_node_trim_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, uint32_t v) {
	reset_data_field<uint32_t>(&ctx->trim, begin, end, v);
}

