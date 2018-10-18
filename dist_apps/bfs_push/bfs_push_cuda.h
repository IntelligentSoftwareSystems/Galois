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

void get_bitset_dist_current_cuda(struct CUDA_Context* ctx,
                                  uint64_t* bitset_compute);
void bitset_dist_current_reset_cuda(struct CUDA_Context* ctx);
void bitset_dist_current_reset_cuda(struct CUDA_Context* ctx, size_t begin,
                                    size_t end);
uint32_t get_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned LID,
                                uint32_t v);
void add_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned LID,
                                uint32_t v);
bool min_node_dist_current_cuda(struct CUDA_Context* ctx, unsigned LID,
                                uint32_t v);
void batch_get_node_dist_current_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint8_t* v);
void batch_get_node_dist_current_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint8_t* v,
                                      size_t* v_size, DataCommMode* data_mode);
void batch_get_mirror_node_dist_current_cuda(struct CUDA_Context* ctx,
                                             unsigned from_id, uint8_t* v);
void batch_get_mirror_node_dist_current_cuda(struct CUDA_Context* ctx,
                                             unsigned from_id, uint8_t* v,
                                             size_t* v_size,
                                             DataCommMode* data_mode);
void batch_get_reset_node_dist_current_cuda(struct CUDA_Context* ctx,
                                            unsigned from_id, uint8_t* v,
                                            uint32_t i);
void batch_get_reset_node_dist_current_cuda(
    struct CUDA_Context* ctx, unsigned from_id,
    uint8_t* v, size_t* v_size, DataCommMode* data_mode,
    uint32_t i);
void batch_set_mirror_node_dist_current_cuda(
    struct CUDA_Context* ctx, unsigned from_id,
    uint8_t* v, DataCommMode data_mode);
void batch_set_node_dist_current_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id,
                                      uint8_t* v,
                                      DataCommMode data_mode);
void batch_add_node_dist_current_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id,
                                      uint8_t* v,
                                      DataCommMode data_mode);
void batch_min_node_dist_current_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id,
                                      uint8_t* v,
                                      DataCommMode data_mode);
void batch_reset_node_dist_current_cuda(struct CUDA_Context* ctx, size_t begin,
                                        size_t end, uint32_t v);

void get_bitset_dist_old_cuda(struct CUDA_Context* ctx,
                              uint64_t* bitset_compute);
void bitset_dist_old_reset_cuda(struct CUDA_Context* ctx);
void bitset_dist_old_reset_cuda(struct CUDA_Context* ctx, size_t begin,
                                size_t end);
uint32_t get_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
void add_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
bool min_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
void batch_get_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                  uint8_t* v);
void batch_get_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                  uint8_t* v, size_t* v_size,
                                  DataCommMode* data_mode);
void batch_get_mirror_node_dist_old_cuda(struct CUDA_Context* ctx,
                                         unsigned from_id, uint8_t* v);
void batch_get_mirror_node_dist_old_cuda(struct CUDA_Context* ctx,
                                         unsigned from_id,
                                         uint8_t* v,
                                         size_t* v_size,
                                         DataCommMode* data_mode);
void batch_get_reset_node_dist_old_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint8_t* v,
                                        uint32_t i);
void batch_get_reset_node_dist_old_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint8_t* v,
                                        size_t* v_size, DataCommMode* data_mode,
                                        uint32_t i);
void batch_set_mirror_node_dist_old_cuda(struct CUDA_Context* ctx,
                                         unsigned from_id,
                                         uint8_t* v,
                                         DataCommMode data_mode);
void batch_set_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                  uint8_t* v,
                                  DataCommMode data_mode);
void batch_add_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                  uint8_t* v,
                                  DataCommMode data_mode);
void batch_min_node_dist_old_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                  uint8_t* v,
                                  DataCommMode data_mode);
void batch_reset_node_dist_old_cuda(struct CUDA_Context* ctx, size_t begin,
                                    size_t end, uint32_t v);

void BFS_cuda(unsigned int __begin, unsigned int __end,
              unsigned int& DGAccumulator_accum, 
              unsigned int& work_items,
              const uint32_t local_priority,
              struct CUDA_Context* ctx);
void BFSSanityCheck_cuda(unsigned int __begin, unsigned int __end,
                         uint64_t& DGAccumulator_sum, uint32_t& DGMax,
                         const uint32_t& local_infinity,
                         struct CUDA_Context* ctx);
void BFSSanityCheck_allNodes_cuda(uint64_t& DGAccumulator_sum, uint32_t& DGMax,
                                  const uint32_t& local_infinity,
                                  struct CUDA_Context* ctx);
void BFSSanityCheck_masterNodes_cuda(uint64_t& DGAccumulator_sum,
                                     uint32_t& DGMax,
                                     const uint32_t& local_infinity,
                                     struct CUDA_Context* ctx);
void BFSSanityCheck_nodesWithEdges_cuda(uint64_t& DGAccumulator_sum,
                                        uint32_t& DGMax,
                                        const uint32_t& local_infinity,
                                        struct CUDA_Context* ctx);
void BFS_allNodes_cuda(unsigned int& DGAccumulator_accum,
                       unsigned int& work_items,
                       const uint32_t local_priority,
                       struct CUDA_Context* ctx);
void BFS_masterNodes_cuda(unsigned int& DGAccumulator_accum,
                          unsigned int& work_items,
                          const uint32_t local_priority,
                          struct CUDA_Context* ctx);
void BFS_nodesWithEdges_cuda(unsigned int& DGAccumulator_accum,
                             unsigned int& work_items,
                             const uint32_t local_priority,
                             struct CUDA_Context* ctx);
void FirstItr_BFS_cuda(unsigned int __begin, unsigned int __end,
                       struct CUDA_Context* ctx);
void FirstItr_BFS_allNodes_cuda(struct CUDA_Context* ctx);
void FirstItr_BFS_masterNodes_cuda(struct CUDA_Context* ctx);
void FirstItr_BFS_nodesWithEdges_cuda(struct CUDA_Context* ctx);
void InitializeGraph_cuda(unsigned int __begin, unsigned int __end,
                          const uint32_t& local_infinity,
                          unsigned long long local_src_node,
                          struct CUDA_Context* ctx);
void InitializeGraph_allNodes_cuda(const uint32_t& local_infinity,
                                   unsigned long long local_src_node,
                                   struct CUDA_Context* ctx);
void InitializeGraph_masterNodes_cuda(const uint32_t& local_infinity,
                                      unsigned long long local_src_node,
                                      struct CUDA_Context* ctx);
void InitializeGraph_nodesWithEdges_cuda(const uint32_t& local_infinity,
                                         unsigned long long local_src_node,
                                         struct CUDA_Context* ctx);
