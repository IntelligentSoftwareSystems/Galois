#pragma once

#include "galois/runtime/DataCommMode.h"
#include "galois/cuda/HostDecls.h"

void get_bitset_delta_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute);
void bitset_delta_reset_cuda(struct CUDA_Context* ctx);
void bitset_delta_reset_cuda(struct CUDA_Context* ctx, size_t begin,
                             size_t end);
float get_node_delta_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_delta_cuda(struct CUDA_Context* ctx, unsigned LID, float v);
void add_node_delta_cuda(struct CUDA_Context* ctx, unsigned LID, float v);
bool min_node_delta_cuda(struct CUDA_Context* ctx, unsigned LID, float v);
void batch_get_node_delta_cuda(struct CUDA_Context* ctx, unsigned from_id,
                               uint8_t* v);
void batch_get_node_delta_cuda(struct CUDA_Context* ctx, unsigned from_id,
                               uint8_t* v, size_t* v_size,
                               DataCommMode* data_mode);
void batch_get_mirror_node_delta_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint8_t* v);
void batch_get_mirror_node_delta_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint8_t* v,
                                      size_t* v_size, DataCommMode* data_mode);
void batch_get_reset_node_delta_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                     uint8_t* v, float i);
void batch_get_reset_node_delta_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                     uint8_t* v, size_t* v_size,
                                     DataCommMode* data_mode, float i);
void batch_set_mirror_node_delta_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint8_t* v,
                                      DataCommMode data_mode);
void batch_set_node_delta_cuda(struct CUDA_Context* ctx, unsigned from_id,
                               uint8_t* v, DataCommMode data_mode);
void batch_add_mirror_node_delta_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint8_t* v,
                                      DataCommMode data_mode);
void batch_add_node_delta_cuda(struct CUDA_Context* ctx, unsigned from_id,
                               uint8_t* v, DataCommMode data_mode);
void batch_min_mirror_node_delta_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint8_t* v,
                                      DataCommMode data_mode);
void batch_min_node_delta_cuda(struct CUDA_Context* ctx, unsigned from_id,
                               uint8_t* v, DataCommMode data_mode);
void batch_reset_node_delta_cuda(struct CUDA_Context* ctx, size_t begin,
                                 size_t end, float v);

void get_bitset_nout_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute);
void bitset_nout_reset_cuda(struct CUDA_Context* ctx);
void bitset_nout_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end);
uint32_t get_node_nout_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_nout_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
void add_node_nout_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
bool min_node_nout_cuda(struct CUDA_Context* ctx, unsigned LID, uint32_t v);
void batch_get_node_nout_cuda(struct CUDA_Context* ctx, unsigned from_id,
                              uint8_t* v);
void batch_get_node_nout_cuda(struct CUDA_Context* ctx, unsigned from_id,
                              uint8_t* v, size_t* v_size,
                              DataCommMode* data_mode);
void batch_get_mirror_node_nout_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                     uint8_t* v);
void batch_get_mirror_node_nout_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                     uint8_t* v, size_t* v_size,
                                     DataCommMode* data_mode);
void batch_get_reset_node_nout_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint8_t* v, uint32_t i);
void batch_get_reset_node_nout_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint8_t* v, size_t* v_size,
                                    DataCommMode* data_mode, uint32_t i);
void batch_set_mirror_node_nout_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                     uint8_t* v, DataCommMode data_mode);
void batch_set_node_nout_cuda(struct CUDA_Context* ctx, unsigned from_id,
                              uint8_t* v, DataCommMode data_mode);
void batch_add_mirror_node_nout_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                     uint8_t* v, DataCommMode data_mode);
void batch_add_node_nout_cuda(struct CUDA_Context* ctx, unsigned from_id,
                              uint8_t* v, DataCommMode data_mode);
void batch_min_mirror_node_nout_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                     uint8_t* v, DataCommMode data_mode);
void batch_min_node_nout_cuda(struct CUDA_Context* ctx, unsigned from_id,
                              uint8_t* v, DataCommMode data_mode);
void batch_reset_node_nout_cuda(struct CUDA_Context* ctx, size_t begin,
                                size_t end, uint32_t v);

void get_bitset_residual_cuda(struct CUDA_Context* ctx,
                              uint64_t* bitset_compute);
void bitset_residual_reset_cuda(struct CUDA_Context* ctx);
void bitset_residual_reset_cuda(struct CUDA_Context* ctx, size_t begin,
                                size_t end);
float get_node_residual_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_residual_cuda(struct CUDA_Context* ctx, unsigned LID, float v);
void add_node_residual_cuda(struct CUDA_Context* ctx, unsigned LID, float v);
bool min_node_residual_cuda(struct CUDA_Context* ctx, unsigned LID, float v);
void batch_get_node_residual_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                  uint8_t* v);
void batch_get_node_residual_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                  uint8_t* v, size_t* v_size,
                                  DataCommMode* data_mode);
void batch_get_mirror_node_residual_cuda(struct CUDA_Context* ctx,
                                         unsigned from_id, uint8_t* v);
void batch_get_mirror_node_residual_cuda(struct CUDA_Context* ctx,
                                         unsigned from_id, uint8_t* v,
                                         size_t* v_size,
                                         DataCommMode* data_mode);
void batch_get_reset_node_residual_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint8_t* v, float i);
void batch_get_reset_node_residual_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint8_t* v,
                                        size_t* v_size, DataCommMode* data_mode,
                                        float i);
void batch_set_mirror_node_residual_cuda(struct CUDA_Context* ctx,
                                         unsigned from_id, uint8_t* v,
                                         DataCommMode data_mode);
void batch_set_node_residual_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                  uint8_t* v, DataCommMode data_mode);
void batch_add_mirror_node_residual_cuda(struct CUDA_Context* ctx,
                                         unsigned from_id, uint8_t* v,
                                         DataCommMode data_mode);
void batch_add_node_residual_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                  uint8_t* v, DataCommMode data_mode);
void batch_min_mirror_node_residual_cuda(struct CUDA_Context* ctx,
                                         unsigned from_id, uint8_t* v,
                                         DataCommMode data_mode);
void batch_min_node_residual_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                  uint8_t* v, DataCommMode data_mode);
void batch_reset_node_residual_cuda(struct CUDA_Context* ctx, size_t begin,
                                    size_t end, float v);

void get_bitset_value_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute);
void bitset_value_reset_cuda(struct CUDA_Context* ctx);
void bitset_value_reset_cuda(struct CUDA_Context* ctx, size_t begin,
                             size_t end);
float get_node_value_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_value_cuda(struct CUDA_Context* ctx, unsigned LID, float v);
void add_node_value_cuda(struct CUDA_Context* ctx, unsigned LID, float v);
bool min_node_value_cuda(struct CUDA_Context* ctx, unsigned LID, float v);
void batch_get_node_value_cuda(struct CUDA_Context* ctx, unsigned from_id,
                               uint8_t* v);
void batch_get_node_value_cuda(struct CUDA_Context* ctx, unsigned from_id,
                               uint8_t* v, size_t* v_size,
                               DataCommMode* data_mode);
void batch_get_mirror_node_value_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint8_t* v);
void batch_get_mirror_node_value_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint8_t* v,
                                      size_t* v_size, DataCommMode* data_mode);
void batch_get_reset_node_value_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                     uint8_t* v, float i);
void batch_get_reset_node_value_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                     uint8_t* v, size_t* v_size,
                                     DataCommMode* data_mode, float i);
void batch_set_mirror_node_value_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint8_t* v,
                                      DataCommMode data_mode);
void batch_set_node_value_cuda(struct CUDA_Context* ctx, unsigned from_id,
                               uint8_t* v, DataCommMode data_mode);
void batch_add_mirror_node_value_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint8_t* v,
                                      DataCommMode data_mode);
void batch_add_node_value_cuda(struct CUDA_Context* ctx, unsigned from_id,
                               uint8_t* v, DataCommMode data_mode);
void batch_min_mirror_node_value_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint8_t* v,
                                      DataCommMode data_mode);
void batch_min_node_value_cuda(struct CUDA_Context* ctx, unsigned from_id,
                               uint8_t* v, DataCommMode data_mode);
void batch_reset_node_value_cuda(struct CUDA_Context* ctx, size_t begin,
                                 size_t end, float v);

void InitializeGraph_cuda(unsigned int __begin, unsigned int __end,
                          struct CUDA_Context* ctx);
void InitializeGraph_allNodes_cuda(struct CUDA_Context* ctx);
void InitializeGraph_masterNodes_cuda(struct CUDA_Context* ctx);
void InitializeGraph_nodesWithEdges_cuda(struct CUDA_Context* ctx);
void PageRank_cuda(unsigned int __begin, unsigned int __end,
                   struct CUDA_Context* ctx);
void PageRankSanity_cuda(unsigned int __begin, unsigned int __end,
                         uint64_t& DGAccumulator_residual_over_tolerance,
                         float& DGAccumulator_sum,
                         float& DGAccumulator_sum_residual, float& max_residual,
                         float& max_value, float& min_residual,
                         float& min_value, float local_tolerance,
                         struct CUDA_Context* ctx);
void PageRankSanity_allNodes_cuda(
    uint64_t& DGAccumulator_residual_over_tolerance, float& DGAccumulator_sum,
    float& DGAccumulator_sum_residual, float& max_residual, float& max_value,
    float& min_residual, float& min_value, float local_tolerance,
    struct CUDA_Context* ctx);
void PageRankSanity_masterNodes_cuda(
    uint64_t& DGAccumulator_residual_over_tolerance, float& DGAccumulator_sum,
    float& DGAccumulator_sum_residual, float& max_residual, float& max_value,
    float& min_residual, float& min_value, float local_tolerance,
    struct CUDA_Context* ctx);
void PageRankSanity_nodesWithEdges_cuda(
    uint64_t& DGAccumulator_residual_over_tolerance, float& DGAccumulator_sum,
    float& DGAccumulator_sum_residual, float& max_residual, float& max_value,
    float& min_residual, float& min_value, float local_tolerance,
    struct CUDA_Context* ctx);
void PageRank_allNodes_cuda(struct CUDA_Context* ctx);
void PageRank_delta_cuda(unsigned int __begin, unsigned int __end,
                         unsigned int& active_vertices,
                         const float& local_alpha, float local_tolerance,
                         struct CUDA_Context* ctx);
void PageRank_delta_allNodes_cuda(unsigned int& active_vertices,
                                  const float& local_alpha,
                                  float local_tolerance,
                                  struct CUDA_Context* ctx);
void PageRank_delta_masterNodes_cuda(unsigned int& active_vertices,
                                     const float& local_alpha,
                                     float local_tolerance,
                                     struct CUDA_Context* ctx);
void PageRank_delta_nodesWithEdges_cuda(unsigned int& active_vertices,
                                        const float& local_alpha,
                                        float local_tolerance,
                                        struct CUDA_Context* ctx);
void PageRank_masterNodes_cuda(struct CUDA_Context* ctx);
void PageRank_nodesWithEdges_cuda(struct CUDA_Context* ctx);
void ResetGraph_cuda(unsigned int __begin, unsigned int __end,
                     const float& local_alpha, struct CUDA_Context* ctx);
void ResetGraph_allNodes_cuda(const float& local_alpha,
                              struct CUDA_Context* ctx);
void ResetGraph_masterNodes_cuda(const float& local_alpha,
                                 struct CUDA_Context* ctx);
void ResetGraph_nodesWithEdges_cuda(const float& local_alpha,
                                    struct CUDA_Context* ctx);
