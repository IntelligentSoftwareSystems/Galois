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

#include "galois/runtime/DataCommMode.h"
#include "galois/cuda/HostDecls.h"

void get_bitset_current_degree_cuda(struct CUDA_Context* ctx,
                                    uint64_t* bitset_compute);
void bitset_current_degree_reset_cuda(struct CUDA_Context* ctx);
void bitset_current_degree_reset_cuda(struct CUDA_Context* ctx, size_t begin,
                                      size_t end);
uint32_t get_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned LID,
                                  uint32_t v);
void add_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned LID,
                                  uint32_t v);
bool min_node_current_degree_cuda(struct CUDA_Context* ctx, unsigned LID,
                                  uint32_t v);
void batch_get_node_current_degree_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint32_t* v);
void batch_get_node_current_degree_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint64_t* bitset_comm,
                                        unsigned int* offsets, uint32_t* v,
                                        size_t* v_size,
                                        DataCommMode* data_mode);
void batch_get_mirror_node_current_degree_cuda(struct CUDA_Context* ctx,
                                               unsigned from_id, uint32_t* v);
void batch_get_mirror_node_current_degree_cuda(struct CUDA_Context* ctx,
                                               unsigned from_id,
                                               uint64_t* bitset_comm,
                                               unsigned int* offsets,
                                               uint32_t* v, size_t* v_size,
                                               DataCommMode* data_mode);
void batch_get_reset_node_current_degree_cuda(struct CUDA_Context* ctx,
                                              unsigned from_id, uint32_t* v,
                                              uint32_t i);
void batch_get_reset_node_current_degree_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint32_t* v, size_t* v_size, DataCommMode* data_mode,
    uint32_t i);
void batch_set_mirror_node_current_degree_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint32_t* v, size_t v_size, DataCommMode data_mode);
void batch_set_node_current_degree_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint64_t* bitset_comm,
                                        unsigned int* offsets, uint32_t* v,
                                        size_t v_size, DataCommMode data_mode);
void batch_add_node_current_degree_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint64_t* bitset_comm,
                                        unsigned int* offsets, uint32_t* v,
                                        size_t v_size, DataCommMode data_mode);
void batch_min_node_current_degree_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint64_t* bitset_comm,
                                        unsigned int* offsets, uint32_t* v,
                                        size_t v_size, DataCommMode data_mode);
void batch_reset_node_current_degree_cuda(struct CUDA_Context* ctx,
                                          size_t begin, size_t end, uint32_t v);

void get_bitset_flag_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute);
void bitset_flag_reset_cuda(struct CUDA_Context* ctx);
void bitset_flag_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end);
uint8_t get_node_flag_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_flag_cuda(struct CUDA_Context* ctx, unsigned LID, uint8_t v);
void add_node_flag_cuda(struct CUDA_Context* ctx, unsigned LID, uint8_t v);
bool min_node_flag_cuda(struct CUDA_Context* ctx, unsigned LID, uint8_t v);
void batch_get_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id,
                              uint8_t* v);
void batch_get_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id,
                              uint64_t* bitset_comm, unsigned int* offsets,
                              uint8_t* v, size_t* v_size,
                              DataCommMode* data_mode);
void batch_get_mirror_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                     uint8_t* v);
void batch_get_mirror_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                     uint64_t* bitset_comm,
                                     unsigned int* offsets, uint8_t* v,
                                     size_t* v_size, DataCommMode* data_mode);
void batch_get_reset_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint8_t* v, uint8_t i);
void batch_get_reset_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint64_t* bitset_comm,
                                    unsigned int* offsets, uint8_t* v,
                                    size_t* v_size, DataCommMode* data_mode,
                                    uint8_t i);
void batch_set_mirror_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                     uint64_t* bitset_comm,
                                     unsigned int* offsets, uint8_t* v,
                                     size_t v_size, DataCommMode data_mode);
void batch_set_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id,
                              uint64_t* bitset_comm, unsigned int* offsets,
                              uint8_t* v, size_t v_size,
                              DataCommMode data_mode);
void batch_add_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id,
                              uint64_t* bitset_comm, unsigned int* offsets,
                              uint8_t* v, size_t v_size,
                              DataCommMode data_mode);
void batch_min_node_flag_cuda(struct CUDA_Context* ctx, unsigned from_id,
                              uint64_t* bitset_comm, unsigned int* offsets,
                              uint8_t* v, size_t v_size,
                              DataCommMode data_mode);
void batch_reset_node_flag_cuda(struct CUDA_Context* ctx, size_t begin,
                                size_t end, uint8_t v);

void get_bitset_trim_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute);
void bitset_trim_reset_cuda(struct CUDA_Context* ctx);
void bitset_trim_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end);
uint32_t get_node_trim_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_trim_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
void add_node_trim_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
bool min_node_trim_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
void batch_get_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id,
                              uint32_t* v);
void batch_get_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id,
                              uint64_t* bitset_comm, unsigned int* offsets,
                              uint32_t* v, size_t* v_size,
                              DataCommMode* data_mode);
void batch_get_mirror_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                     uint32_t* v);
void batch_get_mirror_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                     uint64_t* bitset_comm,
                                     unsigned int* offsets, uint32_t* v,
                                     size_t* v_size, DataCommMode* data_mode);
void batch_get_reset_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint32_t* v, uint32_t i);
void batch_get_reset_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint64_t* bitset_comm,
                                    unsigned int* offsets, uint32_t* v,
                                    size_t* v_size, DataCommMode* data_mode,
                                    uint32_t i);
void batch_set_mirror_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                     uint64_t* bitset_comm,
                                     unsigned int* offsets, uint32_t* v,
                                     size_t v_size, DataCommMode data_mode);
void batch_set_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id,
                              uint64_t* bitset_comm, unsigned int* offsets,
                              uint32_t* v, size_t v_size,
                              DataCommMode data_mode);
void batch_add_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id,
                              uint64_t* bitset_comm, unsigned int* offsets,
                              uint32_t* v, size_t v_size,
                              DataCommMode data_mode);
void batch_min_node_trim_cuda(struct CUDA_Context* ctx, unsigned from_id,
                              uint64_t* bitset_comm, unsigned int* offsets,
                              uint32_t* v, size_t v_size,
                              DataCommMode data_mode);
void batch_reset_node_trim_cuda(struct CUDA_Context* ctx, size_t begin,
                                size_t end, uint32_t v);

void InitializeGraph1_cuda(unsigned int __begin, unsigned int __end,
                           struct CUDA_Context* ctx);
void InitializeGraph1_allNodes_cuda(struct CUDA_Context* ctx);
void InitializeGraph1_masterNodes_cuda(struct CUDA_Context* ctx);
void InitializeGraph1_nodesWithEdges_cuda(struct CUDA_Context* ctx);
void InitializeGraph2_cuda(unsigned int __begin, unsigned int __end,
                           struct CUDA_Context* ctx);
void InitializeGraph2_allNodes_cuda(struct CUDA_Context* ctx);
void InitializeGraph2_masterNodes_cuda(struct CUDA_Context* ctx);
void InitializeGraph2_nodesWithEdges_cuda(struct CUDA_Context* ctx);
void KCoreSanityCheck_cuda(unsigned int __begin, unsigned int __end,
                           uint64_t& DGAccumulator_accum,
                           struct CUDA_Context* ctx);
void KCoreSanityCheck_allNodes_cuda(uint64_t& DGAccumulator_accum,
                                    struct CUDA_Context* ctx);
void KCoreSanityCheck_masterNodes_cuda(uint64_t& DGAccumulator_accum,
                                       struct CUDA_Context* ctx);
void KCoreSanityCheck_nodesWithEdges_cuda(uint64_t& DGAccumulator_accum,
                                          struct CUDA_Context* ctx);
void KCoreStep1_cuda(unsigned int __begin, unsigned int __end,
                     unsigned int& DGAccumulator_accum,
                     uint32_t local_k_core_num, struct CUDA_Context* ctx);
void KCoreStep1_allNodes_cuda(unsigned int& DGAccumulator_accum,
                              uint32_t local_k_core_num,
                              struct CUDA_Context* ctx);
void KCoreStep1_masterNodes_cuda(unsigned int& DGAccumulator_accum,
                                 uint32_t local_k_core_num,
                                 struct CUDA_Context* ctx);
void KCoreStep1_nodesWithEdges_cuda(unsigned int& DGAccumulator_accum,
                                    uint32_t local_k_core_num,
                                    struct CUDA_Context* ctx);
void KCoreStep2_cuda(unsigned int __begin, unsigned int __end,
                     struct CUDA_Context* ctx);
void KCoreStep2_allNodes_cuda(struct CUDA_Context* ctx);
void KCoreStep2_masterNodes_cuda(struct CUDA_Context* ctx);
void KCoreStep2_nodesWithEdges_cuda(struct CUDA_Context* ctx);
