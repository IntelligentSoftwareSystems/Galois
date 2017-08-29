#pragma once
#include <cuda.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include "gen_cuda.h"
#include "Galois/Runtime/Cuda/cuda_helpers.h"

struct CUDA_Context : public CUDA_Context_Common {
	struct CUDA_Context_Field<unsigned int> comp_current;
	struct CUDA_Context_Field<unsigned int> comp_old;
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
	mem_usage += mem_usage_CUDA_field(&ctx->comp_current, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->comp_old, g, num_hosts);
	printf("[%d] Host memory for communication context: %3u MB\n", ctx->id, mem_usage/1048756);
	load_graph_CUDA_common(ctx, g, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->comp_current, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->comp_old, num_hosts);
	reset_CUDA_context(ctx);
}

void reset_CUDA_context(struct CUDA_Context *ctx) {
	ctx->comp_current.data.zero_gpu();
	ctx->comp_old.data.zero_gpu();
}

void bitset_comp_current_clear_cuda(struct CUDA_Context *ctx) {
	ctx->comp_current.is_updated.cpu_rd_ptr()->reset_all();
}

unsigned int get_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID) {
	unsigned int *comp_current = ctx->comp_current.data.cpu_rd_ptr();
	return comp_current[LID];
}

void set_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *comp_current = ctx->comp_current.data.cpu_wr_ptr();
	comp_current[LID] = v;
}

void add_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *comp_current = ctx->comp_current.data.cpu_wr_ptr();
	comp_current[LID] += v;
}

void min_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *comp_current = ctx->comp_current.data.cpu_wr_ptr();
	if (comp_current[LID] > v)
		comp_current[LID] = v;
}

void batch_get_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	batch_get_shared_field<unsigned int, sharedMaster, false>(ctx, &ctx->comp_current, from_id, v);
}

void batch_get_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<unsigned int, sharedMaster, false>(ctx, &ctx->comp_current, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_mirror_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	batch_get_shared_field<unsigned int, sharedMirror, false>(ctx, &ctx->comp_current, from_id, v);
}

void batch_get_mirror_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<unsigned int, sharedMirror, false>(ctx, &ctx->comp_current, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_reset_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i) {
	batch_get_shared_field<unsigned int, sharedMirror, true>(ctx, &ctx->comp_current, from_id, v, i);
}

void batch_get_reset_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode, unsigned int i) {
	batch_get_shared_field<unsigned int, sharedMirror, true>(ctx, &ctx->comp_current, from_id, bitset_comm, offsets, v, v_size, data_mode, i);
}

void batch_set_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<unsigned int, sharedMirror, setOp>(ctx, &ctx->comp_current, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_add_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<unsigned int, sharedMaster, addOp>(ctx, &ctx->comp_current, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_min_node_comp_current_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<unsigned int, sharedMaster, minOp>(ctx, &ctx->comp_current, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void bitset_comp_old_clear_cuda(struct CUDA_Context *ctx) {
	ctx->comp_old.is_updated.cpu_rd_ptr()->reset_all();
}

unsigned int get_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned LID) {
	unsigned int *comp_old = ctx->comp_old.data.cpu_rd_ptr();
	return comp_old[LID];
}

void set_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *comp_old = ctx->comp_old.data.cpu_wr_ptr();
	comp_old[LID] = v;
}

void add_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *comp_old = ctx->comp_old.data.cpu_wr_ptr();
	comp_old[LID] += v;
}

void min_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned LID, unsigned int v) {
	unsigned int *comp_old = ctx->comp_old.data.cpu_wr_ptr();
	if (comp_old[LID] > v)
		comp_old[LID] = v;
}

void batch_get_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	batch_get_shared_field<unsigned int, sharedMaster, false>(ctx, &ctx->comp_old, from_id, v);
}

void batch_get_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<unsigned int, sharedMaster, false>(ctx, &ctx->comp_old, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_mirror_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v) {
	batch_get_shared_field<unsigned int, sharedMirror, false>(ctx, &ctx->comp_old, from_id, v);
}

void batch_get_mirror_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<unsigned int, sharedMirror, false>(ctx, &ctx->comp_old, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_get_reset_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned int *v, unsigned int i) {
	batch_get_shared_field<unsigned int, sharedMirror, true>(ctx, &ctx->comp_old, from_id, v, i);
}

void batch_get_reset_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t *v_size, DataCommMode *data_mode, unsigned int i) {
	batch_get_shared_field<unsigned int, sharedMirror, true>(ctx, &ctx->comp_old, from_id, bitset_comm, offsets, v, v_size, data_mode, i);
}

void batch_set_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<unsigned int, sharedMirror, setOp>(ctx, &ctx->comp_old, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_add_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<unsigned int, sharedMaster, addOp>(ctx, &ctx->comp_old, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

void batch_min_node_comp_old_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, unsigned int *v, size_t v_size, DataCommMode data_mode) {
	batch_set_shared_field<unsigned int, sharedMaster, minOp>(ctx, &ctx->comp_old, from_id, bitset_comm, offsets, v, v_size, data_mode);
}

