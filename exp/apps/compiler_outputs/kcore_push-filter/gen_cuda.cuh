#pragma once
#include <cuda.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include "gen_cuda.h"
#include "Galois/Runtime/Cuda/cuda_helpers.h"

struct CUDA_Context : public CUDA_Context_Common {
	struct CUDA_Context_Field<unsigned int> current_degree;
	struct CUDA_Context_Field<bool> flag;
	struct CUDA_Context_Field<unsigned int> trim;
};

struct CUDA_Context *get_CUDA_context(int id) {
	struct CUDA_Context *ctx;
	ctx = (struct CUDA_Context *) calloc(1, sizeof(struct CUDA_Context));
	ctx->id = id;
	return ctx;
}

bool init_CUDA_context(struct CUDA_Context *ctx, int device) {
	return init_CUDA_context_common(ctx, device);
}

void load_graph_CUDA(struct CUDA_Context *ctx, MarshalGraph &g, unsigned num_hosts) {
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

void reset_CUDA_context(struct CUDA_Context *ctx) {
	ctx->current_degree.data.zero_gpu();
	ctx->flag.data.zero_gpu();
	ctx->trim.data.zero_gpu();
}

void bitset_current_degree_clear_cuda(struct CUDA_Context *ctx) {
	ctx->current_degree.is_updated.cpu_rd_ptr()->reset_all();
}

unsigned int get_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned LID) {
	unsigned int *current_degree = ctx->current_degree.data.cpu_rd_ptr();
	return current_degree[LID];
}

void set_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *current_degree = ctx->current_degree.data.cpu_wr_ptr();
	current_degree[LID] = v;
}

void add_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *current_degree = ctx->current_degree.data.cpu_wr_ptr();
	current_degree[LID] += v;
}

void min_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *current_degree = ctx->current_degree.data.cpu_wr_ptr();
	if (current_degree[LID] > v)
		current_degree[LID] = v;
}

void batch_get_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	batch_get_shared_field<unsigned int, sharedMaster, false>(ctx, &ctx->current_degree, from_id, v);
}

void batch_get_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<unsigned int, sharedMaster, false>(ctx, &ctx->current_degree, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_mirror_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	batch_get_shared_field<unsigned int, sharedMirror, false>(ctx, &ctx->current_degree, from_id, v);
}

void batch_get_mirror_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<unsigned int, sharedMirror, false>(ctx, &ctx->current_degree, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_reset_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i) {
	batch_get_shared_field<unsigned int, sharedMirror, true>(ctx, &ctx->current_degree, from_id, v, i);
}

void batch_get_reset_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode, unsigned int i) {
	batch_get_shared_field<unsigned int, sharedMirror, true>(ctx, &ctx->current_degree, from_id, bitset_comm, offsets, v, v_size, data_mode, i);
}

void batch_set_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<unsigned int, sharedMirror, setOp>(ctx, &ctx->current_degree, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_add_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<unsigned int, sharedMaster, addOp>(ctx, &ctx->current_degree, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_min_node_current_degree_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<unsigned int, sharedMaster, minOp>(ctx, &ctx->current_degree, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void bitset_flag_clear_cuda(struct CUDA_Context *ctx) {
	ctx->flag.is_updated.cpu_rd_ptr()->reset_all();
}

bool get_node_flag_cuda(struct CUDA_Context *ctx, unsigned LID) {
	bool *flag = ctx->flag.data.cpu_rd_ptr();
	return flag[LID];
}

void set_node_flag_cuda(struct CUDA_Context *ctx, unsigned LID, bool v) {
	bool *flag = ctx->flag.data.cpu_wr_ptr();
	flag[LID] = v;
}

void add_node_flag_cuda(struct CUDA_Context *ctx, unsigned LID, bool v) {
	bool *flag = ctx->flag.data.cpu_wr_ptr();
	flag[LID] += v;
}

void min_node_flag_cuda(struct CUDA_Context *ctx, unsigned LID, bool v) {
	bool *flag = ctx->flag.data.cpu_wr_ptr();
	if (flag[LID] > v)
		flag[LID] = v;
}

void batch_get_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, bool *v) {
	batch_get_shared_field<bool, sharedMaster, false>(ctx, &ctx->flag, from_id, v);
}

void batch_get_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<bool, sharedMaster, false>(ctx, &ctx->flag, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_mirror_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, bool *v) {
	batch_get_shared_field<bool, sharedMirror, false>(ctx, &ctx->flag, from_id, v);
}

void batch_get_mirror_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<bool, sharedMirror, false>(ctx, &ctx->flag, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_reset_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, bool *v, bool i) {
	batch_get_shared_field<bool, sharedMirror, true>(ctx, &ctx->flag, from_id, v, i);
}

void batch_get_reset_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t *v_size, DataCommMode *data_mode, bool i) {
	batch_get_shared_field<bool, sharedMirror, true>(ctx, &ctx->flag, from_id, bitset_comm, offsets, v, v_size, data_mode, i);
}

void batch_set_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<bool, sharedMirror, setOp>(ctx, &ctx->flag, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_add_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<bool, sharedMaster, addOp>(ctx, &ctx->flag, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_min_node_flag_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, bool *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<bool, sharedMaster, minOp>(ctx, &ctx->flag, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void bitset_trim_clear_cuda(struct CUDA_Context *ctx) {
	ctx->trim.is_updated.cpu_rd_ptr()->reset_all();
}

unsigned int get_node_trim_cuda(struct CUDA_Context *ctx, unsigned LID) {
	unsigned int *trim = ctx->trim.data.cpu_rd_ptr();
	return trim[LID];
}

void set_node_trim_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *trim = ctx->trim.data.cpu_wr_ptr();
	trim[LID] = v;
}

void add_node_trim_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *trim = ctx->trim.data.cpu_wr_ptr();
	trim[LID] += v;
}

void min_node_trim_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *trim = ctx->trim.data.cpu_wr_ptr();
	if (trim[LID] > v)
		trim[LID] = v;
}

void batch_get_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	batch_get_shared_field<unsigned int, sharedMaster, false>(ctx, &ctx->trim, from_id, v);
}

void batch_get_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<unsigned int, sharedMaster, false>(ctx, &ctx->trim, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_mirror_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	batch_get_shared_field<unsigned int, sharedMirror, false>(ctx, &ctx->trim, from_id, v);
}

void batch_get_mirror_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<unsigned int, sharedMirror, false>(ctx, &ctx->trim, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_reset_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i) {
	batch_get_shared_field<unsigned int, sharedMirror, true>(ctx, &ctx->trim, from_id, v, i);
}

void batch_get_reset_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode, unsigned int i) {
	batch_get_shared_field<unsigned int, sharedMirror, true>(ctx, &ctx->trim, from_id, bitset_comm, offsets, v, v_size, data_mode, i);
}

void batch_set_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<unsigned int, sharedMirror, setOp>(ctx, &ctx->trim, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_add_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<unsigned int, sharedMaster, addOp>(ctx, &ctx->trim, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_min_node_trim_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<unsigned int, sharedMaster, minOp>(ctx, &ctx->trim, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

