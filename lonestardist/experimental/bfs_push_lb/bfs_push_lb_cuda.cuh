/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#pragma once
#include <cuda.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include "bfs_push_lb_cuda.h"
#include "galois/runtime/cuda/DeviceSync.h"

struct CUDA_Stat_Context {
	Shared<uint32_t> thread_blocks_work;
	Shared<clock_t> thread_clock_cycles;
	Shared<clock_t> thread_overhead_cycles;
	PipeContextT<Worklist2> thread_work_wl;
	PipeContextT<Worklist2> thread_src_wl;
	Shared<uint32_t> thread_prefix_work_wl;
};


struct CUDA_Context : public CUDA_Context_Common {
	struct CUDA_Context_Field<uint32_t> dist_current;
	struct CUDA_Context_Field<uint32_t> dist_old;
	struct CUDA_Stat_Context stats;
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
	mem_usage += mem_usage_CUDA_field(&ctx->dist_current, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->dist_old, g, num_hosts);
	printf("[%d] Host memory for communication context: %3u MB\n", ctx->id, mem_usage/1048756);
	load_graph_CUDA_common(ctx, g, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->dist_current, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->dist_old, num_hosts);
	reset_CUDA_context(ctx);
	init_CUDA_stat_context(ctx);
}

void init_CUDA_stat_context(struct CUDA_Context* ctx){

	ctx->stats.thread_work_wl = PipeContextT<Worklist2>(ctx->numNodesWithEdges);
	ctx->stats.thread_src_wl = PipeContextT<Worklist2>(ctx->numNodesWithEdges);

	ctx->stats.thread_prefix_work_wl.alloc(ctx->numNodesWithEdges);
	ctx->stats.thread_prefix_work_wl.zero_gpu();
}

void reset_CUDA_context(struct CUDA_Context* ctx) {
	ctx->dist_current.data.zero_gpu();
	ctx->dist_old.data.zero_gpu();
}

void get_bitset_dist_current_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->dist_current.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_dist_current_reset_cuda(struct CUDA_Context* ctx) {
	ctx->dist_current.is_updated.cpu_rd_ptr()->reset();
}

void bitset_dist_current_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->dist_current, begin, end);
}

uint32_t get_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned LID) {
	uint32_t *dist_current = ctx->dist_current.data.cpu_rd_ptr();
	return dist_current[LID];
}

void set_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *dist_current = ctx->dist_current.data.cpu_wr_ptr();
	dist_current[LID] = v;
}

void add_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *dist_current = ctx->dist_current.data.cpu_wr_ptr();
	dist_current[LID] += v;
}

bool min_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *dist_current = ctx->dist_current.data.cpu_wr_ptr();
	if (dist_current[LID] > v){
		dist_current[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->dist_current, from_id, v);
}

void batch_get_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->dist_current, from_id, v, v_size, data_mode);
}

void batch_get_mirror_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->dist_current, from_id, v);
}

void batch_get_mirror_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->dist_current, from_id, v, v_size, data_mode);
}

void batch_get_reset_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->dist_current, from_id, v, i);
}

void batch_get_reset_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->dist_current, from_id, v, v_size, data_mode, i);
}

void batch_set_mirror_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, setOp>(ctx, &ctx->dist_current, from_id, v, data_mode);
}

void batch_set_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, setOp>(ctx, &ctx->dist_current, from_id, v, data_mode);
}

void batch_add_mirror_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, addOp>(ctx, &ctx->dist_current, from_id, v, data_mode);
}

void batch_add_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, addOp>(ctx, &ctx->dist_current, from_id, v, data_mode);
}

void batch_min_mirror_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, minOp>(ctx, &ctx->dist_current, from_id, v, data_mode);
}

void batch_min_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, minOp>(ctx, &ctx->dist_current, from_id, v, data_mode);
}

void batch_reset_node_dist_current_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, uint32_t v) {
	reset_data_field<uint32_t>(&ctx->dist_current, begin, end, v);
}

void get_bitset_dist_old_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
	ctx->dist_old.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_dist_old_reset_cuda(struct CUDA_Context* ctx) {
	ctx->dist_old.is_updated.cpu_rd_ptr()->reset();
}

void bitset_dist_old_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
	reset_bitset_field(&ctx->dist_old, begin, end);
}

uint32_t get_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned LID) {
	uint32_t *dist_old = ctx->dist_old.data.cpu_rd_ptr();
	return dist_old[LID];
}

void set_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *dist_old = ctx->dist_old.data.cpu_wr_ptr();
	dist_old[LID] = v;
}

void add_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *dist_old = ctx->dist_old.data.cpu_wr_ptr();
	dist_old[LID] += v;
}

bool min_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v) {
	uint32_t *dist_old = ctx->dist_old.data.cpu_wr_ptr();
	if (dist_old[LID] > v){
		dist_old[LID] = v;
		return true;
	}
	return false;
}

void batch_get_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->dist_old, from_id, v);
}

void batch_get_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->dist_old, from_id, v, v_size, data_mode);
}

void batch_get_mirror_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->dist_old, from_id, v);
}

void batch_get_mirror_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->dist_old, from_id, v, v_size, data_mode);
}

void batch_get_reset_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->dist_old, from_id, v, i);
}

void batch_get_reset_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->dist_old, from_id, v, v_size, data_mode, i);
}

void batch_set_mirror_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, setOp>(ctx, &ctx->dist_old, from_id, v, data_mode);
}

void batch_set_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, setOp>(ctx, &ctx->dist_old, from_id, v, data_mode);
}

void batch_add_mirror_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, addOp>(ctx, &ctx->dist_old, from_id, v, data_mode);
}

void batch_add_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, addOp>(ctx, &ctx->dist_old, from_id, v, data_mode);
}

void batch_min_mirror_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, minOp>(ctx, &ctx->dist_old, from_id, v, data_mode);
}

void batch_min_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, minOp>(ctx, &ctx->dist_old, from_id, v, data_mode);
}

void batch_reset_node_dist_old_cuda(struct CUDA_Context* ctx, size_t begin, size_t end, uint32_t v) {
	reset_data_field<uint32_t>(&ctx->dist_old, begin, end, v);
}

