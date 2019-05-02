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
#include "gen_cuda.h"
#include "galois/runtime/cuda/DeviceSync.h"

struct CUDA_Context : public CUDA_Context_Common {
	struct CUDA_Context_Field<float> delta;
	struct CUDA_Context_Field<uint32_t> nout;
	struct CUDA_Context_Field<float> residual;
	struct CUDA_Context_Field<float> value;
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
	mem_usage += mem_usage_CUDA_field(&ctx->delta, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->nout, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->residual, g, num_hosts);
	mem_usage += mem_usage_CUDA_field(&ctx->value, g, num_hosts);
	printf("[%d] Host memory for communication context: %3u MB\n", ctx->id, mem_usage/1048756);
	load_graph_CUDA_common(ctx, g, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->delta, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->nout, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->residual, num_hosts);
	load_graph_CUDA_field(ctx, &ctx->value, num_hosts);
	reset_CUDA_context(ctx);
}

void reset_CUDA_context(struct CUDA_Context *ctx) {
	ctx->delta.data.zero_gpu();
	ctx->nout.data.zero_gpu();
	ctx->residual.data.zero_gpu();
	ctx->value.data.zero_gpu();
}

float get_node_delta_cuda(struct CUDA_Context *ctx, unsigned LID) {
	float *delta = ctx->delta.data.cpu_rd_ptr();
	return delta[LID];
}

void set_node_delta_cuda(struct CUDA_Context *ctx, unsigned LID, float v) {
	float *delta = ctx->delta.data.cpu_wr_ptr();
	delta[LID] = v;
}

void add_node_delta_cuda(struct CUDA_Context *ctx, unsigned LID, float v) {
	float *delta = ctx->delta.data.cpu_wr_ptr();
	delta[LID] += v;
}

bool min_node_delta_cuda(struct CUDA_Context *ctx, unsigned LID, float v) {
	float *delta = ctx->delta.data.cpu_wr_ptr();
	if (delta[LID] > v) {
		delta[LID] = v;
    return true;
  }
  return false;
}

void batch_get_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->delta, from_id, v);
}

void batch_get_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->delta, from_id, v, v_size, data_mode);
}

void batch_get_mirror_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->delta, from_id, v);
}

void batch_get_mirror_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->delta, from_id, v, v_size, data_mode);
}

void batch_get_reset_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->delta, from_id, v, i);
}

void batch_get_reset_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->delta, from_id, v, v_size, data_mode, i);
}

void batch_set_mirror_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMirror, setOp>(ctx, &ctx->delta, from_id, v, data_mode);
}

void batch_set_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, setOp>(ctx, &ctx->delta, from_id, v, data_mode);
}

void batch_add_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, addOp>(ctx, &ctx->delta, from_id, v, data_mode);
}

void batch_min_node_delta_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, minOp>(ctx, &ctx->delta, from_id, v, data_mode);
}

void get_bitset_nout_cuda(struct CUDA_Context *ctx, unsigned long long int *bitset_compute) {
	ctx->nout.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_nout_reset_cuda(struct CUDA_Context *ctx) {
	ctx->nout.is_updated.cpu_rd_ptr()->reset();
}

void bitset_nout_reset_cuda(struct CUDA_Context *ctx, size_t begin, size_t end) {
  reset_bitset_field(&ctx->nout, begin, end);
}

uint32_t get_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID) {
	uint32_t *nout = ctx->nout.data.cpu_rd_ptr();
	return nout[LID];
}

void set_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v) {
	uint32_t *nout = ctx->nout.data.cpu_wr_ptr();
	nout[LID] = v;
}

void add_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v) {
	uint32_t *nout = ctx->nout.data.cpu_wr_ptr();
	nout[LID] += v;
}

bool min_node_nout_cuda(struct CUDA_Context *ctx, unsigned LID, uint32_t v) {
	uint32_t *nout = ctx->nout.data.cpu_wr_ptr();
	if (nout[LID] > v) {
		nout[LID] = v;
    return true;
  }
  return false;
}

void batch_get_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->nout, from_id, v);
}

void batch_get_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint32_t, sharedMaster, false>(ctx, &ctx->nout, from_id, v, v_size, data_mode);
}

void batch_get_mirror_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->nout, from_id, v);
}

void batch_get_mirror_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<uint32_t, sharedMirror, false>(ctx, &ctx->nout, from_id, v, v_size, data_mode);
}

void batch_get_reset_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->nout, from_id, v, i);
}

void batch_get_reset_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode, uint32_t i) {
	batch_get_shared_field<uint32_t, sharedMirror, true>(ctx, &ctx->nout, from_id, v, v_size, data_mode, i);
}

void batch_set_mirror_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMirror, setOp>(ctx, &ctx->nout, from_id, v, data_mode);
}

void batch_set_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, setOp>(ctx, &ctx->nout, from_id, v, data_mode);
}

void batch_add_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, addOp>(ctx, &ctx->nout, from_id, v, data_mode);
}

void batch_min_node_nout_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<uint32_t, sharedMaster, minOp>(ctx, &ctx->nout, from_id, v, data_mode);
}

void get_bitset_residual_cuda(struct CUDA_Context *ctx, unsigned long long int *bitset_compute) {
	ctx->residual.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_residual_reset_cuda(struct CUDA_Context *ctx) {
	ctx->residual.is_updated.cpu_rd_ptr()->reset();
}

void bitset_residual_reset_cuda(struct CUDA_Context *ctx, size_t begin, size_t end) {
  reset_bitset_field(&ctx->residual, begin, end);
}

float get_node_residual_cuda(struct CUDA_Context *ctx, unsigned LID) {
	float *residual = ctx->residual.data.cpu_rd_ptr();
	return residual[LID];
}

void set_node_residual_cuda(struct CUDA_Context *ctx, unsigned LID, float v) {
	float *residual = ctx->residual.data.cpu_wr_ptr();
	residual[LID] = v;
}

void add_node_residual_cuda(struct CUDA_Context *ctx, unsigned LID, float v) {
	float *residual = ctx->residual.data.cpu_wr_ptr();
	residual[LID] += v;
}

bool min_node_residual_cuda(struct CUDA_Context *ctx, unsigned LID, float v) {
	float *residual = ctx->residual.data.cpu_wr_ptr();
	if (residual[LID] > v) {
		residual[LID] = v;
    return true;
  }
  return false;
}

void batch_get_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->residual, from_id, v);
}

void batch_get_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->residual, from_id, v, v_size, data_mode);
}

void batch_get_mirror_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->residual, from_id, v);
}

void batch_get_mirror_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->residual, from_id, v, v_size, data_mode);
}

void batch_get_reset_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->residual, from_id, v, i);
}

void batch_get_reset_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->residual, from_id, v, v_size, data_mode, i);
}

void batch_set_mirror_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMirror, setOp>(ctx, &ctx->residual, from_id, v, data_mode);
}

void batch_set_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, setOp>(ctx, &ctx->residual, from_id, v, data_mode);
}

void batch_add_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, addOp>(ctx, &ctx->residual, from_id, v, data_mode);
}

void batch_min_node_residual_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, minOp>(ctx, &ctx->residual, from_id, v, data_mode);
}

void bitset_value_clear_cuda(struct CUDA_Context *ctx) {
	ctx->value.is_updated.cpu_rd_ptr()->reset();
}

float get_node_value_cuda(struct CUDA_Context *ctx, unsigned LID) {
	float *value = ctx->value.data.cpu_rd_ptr();
	return value[LID];
}

void set_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v) {
	float *value = ctx->value.data.cpu_wr_ptr();
	value[LID] = v;
}

void add_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v) {
	float *value = ctx->value.data.cpu_wr_ptr();
	value[LID] += v;
}

bool min_node_value_cuda(struct CUDA_Context *ctx, unsigned LID, float v) {
	float *value = ctx->value.data.cpu_wr_ptr();
	if (value[LID] > v) {
		value[LID] = v;
    return true;
  }
  return false;
}

void batch_get_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->value, from_id, v);
}

void batch_get_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<float, sharedMaster, false>(ctx, &ctx->value, from_id, v, v_size, data_mode);
}

void batch_get_mirror_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->value, from_id, v);
}

void batch_get_mirror_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode) {
	batch_get_shared_field<float, sharedMirror, false>(ctx, &ctx->value, from_id, v, v_size, data_mode);
}

void batch_get_reset_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->value, from_id, v, i);
}

void batch_get_reset_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, size_t *v_size, DataCommMode *data_mode, float i) {
	batch_get_shared_field<float, sharedMirror, true>(ctx, &ctx->value, from_id, v, v_size, data_mode, i);
}

void batch_set_mirror_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMirror, setOp>(ctx, &ctx->value, from_id, v, data_mode);
}

void batch_set_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, setOp>(ctx, &ctx->value, from_id, v, data_mode);
}

void batch_add_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, addOp>(ctx, &ctx->value, from_id, v, data_mode);
}

void batch_min_node_value_cuda(struct CUDA_Context *ctx, unsigned from_id, uint8_t *v, DataCommMode data_mode) {
	batch_set_shared_field<float, sharedMaster, minOp>(ctx, &ctx->value, from_id, v, data_mode);
}

