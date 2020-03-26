#pragma once

#include "galois/runtime/DataCommMode.h"
#include "galois/cuda/HostDecls.h"

// type of the num shortest paths variable
using ShortPathType = double;

void get_bitset_betweeness_centrality_cuda(struct CUDA_Context* ctx,
                                           uint64_t* bitset_compute);
void bitset_betweeness_centrality_reset_cuda(struct CUDA_Context* ctx);
void bitset_betweeness_centrality_reset_cuda(struct CUDA_Context* ctx,
                                             size_t begin, size_t end);
float get_node_betweeness_centrality_cuda(struct CUDA_Context* ctx,
                                          unsigned LID);
void set_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned LID,
                                         float v);
void add_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned LID,
                                         float v);
bool min_node_betweeness_centrality_cuda(struct CUDA_Context* ctx, unsigned LID,
                                         float v);
void batch_get_node_betweeness_centrality_cuda(struct CUDA_Context* ctx,
                                               unsigned from_id, uint8_t* v);
void batch_get_node_betweeness_centrality_cuda(struct CUDA_Context* ctx,
                                               unsigned from_id, uint8_t* v,
                                               size_t* v_size,
                                               DataCommMode* data_mode);
void batch_get_mirror_node_betweeness_centrality_cuda(struct CUDA_Context* ctx,
                                                      unsigned from_id,
                                                      uint8_t* v);
void batch_get_mirror_node_betweeness_centrality_cuda(struct CUDA_Context* ctx,
                                                      unsigned from_id,
                                                      uint8_t* v,
                                                      size_t* v_size,
                                                      DataCommMode* data_mode);
void batch_get_reset_node_betweeness_centrality_cuda(struct CUDA_Context* ctx,
                                                     unsigned from_id,
                                                     uint8_t* v, float i);
void batch_get_reset_node_betweeness_centrality_cuda(struct CUDA_Context* ctx,
                                                     unsigned from_id,
                                                     uint8_t* v, size_t* v_size,
                                                     DataCommMode* data_mode,
                                                     float i);
void batch_set_mirror_node_betweeness_centrality_cuda(struct CUDA_Context* ctx,
                                                      unsigned from_id,
                                                      uint8_t* v,
                                                      DataCommMode data_mode);
void batch_set_node_betweeness_centrality_cuda(struct CUDA_Context* ctx,
                                               unsigned from_id, uint8_t* v,
                                               DataCommMode data_mode);
void batch_add_mirror_node_betweeness_centrality_cuda(struct CUDA_Context* ctx,
                                                      unsigned from_id,
                                                      uint8_t* v,
                                                      DataCommMode data_mode);
void batch_add_node_betweeness_centrality_cuda(struct CUDA_Context* ctx,
                                               unsigned from_id, uint8_t* v,
                                               DataCommMode data_mode);
void batch_min_mirror_node_betweeness_centrality_cuda(struct CUDA_Context* ctx,
                                                      unsigned from_id,
                                                      uint8_t* v,
                                                      DataCommMode data_mode);
void batch_min_node_betweeness_centrality_cuda(struct CUDA_Context* ctx,
                                               unsigned from_id, uint8_t* v,
                                               DataCommMode data_mode);
void batch_reset_node_betweeness_centrality_cuda(struct CUDA_Context* ctx,
                                                 size_t begin, size_t end,
                                                 float v);

void get_bitset_current_length_cuda(struct CUDA_Context* ctx,
                                    uint64_t* bitset_compute);
void bitset_current_length_reset_cuda(struct CUDA_Context* ctx);
void bitset_current_length_reset_cuda(struct CUDA_Context* ctx, size_t begin,
                                      size_t end);
uint32_t get_node_current_length_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_current_length_cuda(struct CUDA_Context* ctx, unsigned LID,
                                  uint32_t v);
void add_node_current_length_cuda(struct CUDA_Context* ctx, unsigned LID,
                                  uint32_t v);
bool min_node_current_length_cuda(struct CUDA_Context* ctx, unsigned LID,
                                  uint32_t v);
void batch_get_node_current_length_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint8_t* v);
void batch_get_node_current_length_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint8_t* v,
                                        size_t* v_size,
                                        DataCommMode* data_mode);
void batch_get_mirror_node_current_length_cuda(struct CUDA_Context* ctx,
                                               unsigned from_id, uint8_t* v);
void batch_get_mirror_node_current_length_cuda(struct CUDA_Context* ctx,
                                               unsigned from_id, uint8_t* v,
                                               size_t* v_size,
                                               DataCommMode* data_mode);
void batch_get_reset_node_current_length_cuda(struct CUDA_Context* ctx,
                                              unsigned from_id, uint8_t* v,
                                              uint32_t i);
void batch_get_reset_node_current_length_cuda(struct CUDA_Context* ctx,
                                              unsigned from_id, uint8_t* v,
                                              size_t* v_size,
                                              DataCommMode* data_mode,
                                              uint32_t i);
void batch_set_mirror_node_current_length_cuda(struct CUDA_Context* ctx,
                                               unsigned from_id, uint8_t* v,
                                               DataCommMode data_mode);
void batch_set_node_current_length_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint8_t* v,
                                        DataCommMode data_mode);
void batch_add_mirror_node_current_length_cuda(struct CUDA_Context* ctx,
                                               unsigned from_id, uint8_t* v,
                                               DataCommMode data_mode);
void batch_add_node_current_length_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint8_t* v,
                                        DataCommMode data_mode);
void batch_min_mirror_node_current_length_cuda(struct CUDA_Context* ctx,
                                               unsigned from_id, uint8_t* v,
                                               DataCommMode data_mode);
void batch_min_node_current_length_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint8_t* v,
                                        DataCommMode data_mode);
void batch_reset_node_current_length_cuda(struct CUDA_Context* ctx,
                                          size_t begin, size_t end, uint32_t v);

void get_bitset_dependency_cuda(struct CUDA_Context* ctx,
                                uint64_t* bitset_compute);
void bitset_dependency_reset_cuda(struct CUDA_Context* ctx);
void bitset_dependency_reset_cuda(struct CUDA_Context* ctx, size_t begin,
                                  size_t end);
float get_node_dependency_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_dependency_cuda(struct CUDA_Context* ctx, unsigned LID, float v);
void add_node_dependency_cuda(struct CUDA_Context* ctx, unsigned LID, float v);
bool min_node_dependency_cuda(struct CUDA_Context* ctx, unsigned LID, float v);
void batch_get_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint8_t* v);
void batch_get_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint8_t* v, size_t* v_size,
                                    DataCommMode* data_mode);
void batch_get_mirror_node_dependency_cuda(struct CUDA_Context* ctx,
                                           unsigned from_id, uint8_t* v);
void batch_get_mirror_node_dependency_cuda(struct CUDA_Context* ctx,
                                           unsigned from_id, uint8_t* v,
                                           size_t* v_size,
                                           DataCommMode* data_mode);
void batch_get_reset_node_dependency_cuda(struct CUDA_Context* ctx,
                                          unsigned from_id, uint8_t* v,
                                          float i);
void batch_get_reset_node_dependency_cuda(struct CUDA_Context* ctx,
                                          unsigned from_id, uint8_t* v,
                                          size_t* v_size,
                                          DataCommMode* data_mode, float i);
void batch_set_mirror_node_dependency_cuda(struct CUDA_Context* ctx,
                                           unsigned from_id, uint8_t* v,
                                           DataCommMode data_mode);
void batch_set_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint8_t* v, DataCommMode data_mode);
void batch_add_mirror_node_dependency_cuda(struct CUDA_Context* ctx,
                                           unsigned from_id, uint8_t* v,
                                           DataCommMode data_mode);
void batch_add_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint8_t* v, DataCommMode data_mode);
void batch_min_mirror_node_dependency_cuda(struct CUDA_Context* ctx,
                                           unsigned from_id, uint8_t* v,
                                           DataCommMode data_mode);
void batch_min_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint8_t* v, DataCommMode data_mode);
void batch_reset_node_dependency_cuda(struct CUDA_Context* ctx, size_t begin,
                                      size_t end, float v);

void get_bitset_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                        uint64_t* bitset_compute);
void bitset_num_shortest_paths_reset_cuda(struct CUDA_Context* ctx);
void bitset_num_shortest_paths_reset_cuda(struct CUDA_Context* ctx,
                                          size_t begin, size_t end);
ShortPathType get_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                               unsigned LID);
void set_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned LID,
                                      ShortPathType v);
void add_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned LID,
                                      ShortPathType v);
bool min_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned LID,
                                      ShortPathType v);
void batch_get_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                            unsigned from_id, uint8_t* v);
void batch_get_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                            unsigned from_id, uint8_t* v,
                                            size_t* v_size,
                                            DataCommMode* data_mode);
void batch_get_mirror_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                                   unsigned from_id,
                                                   uint8_t* v);
void batch_get_mirror_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                                   unsigned from_id, uint8_t* v,
                                                   size_t* v_size,
                                                   DataCommMode* data_mode);
void batch_get_reset_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                                  unsigned from_id, uint8_t* v,
                                                  ShortPathType i);
void batch_get_reset_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                                  unsigned from_id, uint8_t* v,
                                                  size_t* v_size,
                                                  DataCommMode* data_mode,
                                                  ShortPathType i);
void batch_set_mirror_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                                   unsigned from_id, uint8_t* v,
                                                   DataCommMode data_mode);
void batch_set_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                            unsigned from_id, uint8_t* v,
                                            DataCommMode data_mode);
void batch_add_mirror_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                                   unsigned from_id, uint8_t* v,
                                                   DataCommMode data_mode);
void batch_add_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                            unsigned from_id, uint8_t* v,
                                            DataCommMode data_mode);
void batch_min_mirror_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                                   unsigned from_id, uint8_t* v,
                                                   DataCommMode data_mode);
void batch_min_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                            unsigned from_id, uint8_t* v,
                                            DataCommMode data_mode);
void batch_reset_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                              size_t begin, size_t end,
                                              ShortPathType v);

void BC_cuda(unsigned int __begin, unsigned int __end,
             struct CUDA_Context* ctx);
void BC_allNodes_cuda(struct CUDA_Context* ctx);
void BC_masterNodes_cuda(struct CUDA_Context* ctx);
void BC_nodesWithEdges_cuda(struct CUDA_Context* ctx);
void BackwardPass_cuda(unsigned int __begin, unsigned int __end,
                       uint32_t local_r, struct CUDA_Context* ctx);
void BackwardPass_allNodes_cuda(uint32_t local_r, struct CUDA_Context* ctx);
void BackwardPass_masterNodes_cuda(uint32_t local_r, struct CUDA_Context* ctx);
void BackwardPass_nodesWithEdges_cuda(uint32_t local_r,
                                      struct CUDA_Context* ctx);
void ForwardPass_cuda(unsigned int __begin, unsigned int __end, uint32_t& dga,
                      uint32_t local_r, struct CUDA_Context* ctx);
void ForwardPass_allNodes_cuda(uint32_t& dga, uint32_t local_r,
                               struct CUDA_Context* ctx);
void ForwardPass_masterNodes_cuda(uint32_t& dga, uint32_t local_r,
                                  struct CUDA_Context* ctx);
void ForwardPass_nodesWithEdges_cuda(uint32_t& dga, uint32_t local_r,
                                     struct CUDA_Context* ctx);
void InitializeGraph_cuda(unsigned int __begin, unsigned int __end,
                          struct CUDA_Context* ctx);
void InitializeGraph_allNodes_cuda(struct CUDA_Context* ctx);
void InitializeGraph_masterNodes_cuda(struct CUDA_Context* ctx);
void InitializeGraph_nodesWithEdges_cuda(struct CUDA_Context* ctx);
void InitializeIteration_cuda(unsigned int __begin, unsigned int __end,
                              const uint32_t& local_infinity,
                              const uint64_t& local_current_src_node,
                              struct CUDA_Context* ctx);
void InitializeIteration_allNodes_cuda(const uint32_t& local_infinity,
                                       const uint64_t& local_current_src_node,
                                       struct CUDA_Context* ctx);
void InitializeIteration_masterNodes_cuda(
    const uint32_t& local_infinity, const uint64_t& local_current_src_node,
    struct CUDA_Context* ctx);
void InitializeIteration_nodesWithEdges_cuda(
    const uint32_t& local_infinity, const uint64_t& local_current_src_node,
    struct CUDA_Context* ctx);
void MiddleSync_cuda(unsigned int __begin, unsigned int __end,
                     const uint32_t local_infinity, struct CUDA_Context* ctx);
void MiddleSync_allNodes_cuda(const uint32_t local_infinity,
                              struct CUDA_Context* ctx);
void MiddleSync_masterNodes_cuda(const uint32_t local_infinity,
                                 struct CUDA_Context* ctx);
void MiddleSync_nodesWithEdges_cuda(const uint32_t local_infinity,
                                    struct CUDA_Context* ctx);
void Sanity_cuda(unsigned int __begin, unsigned int __end,
                 float& DGAccumulator_sum, float& DGAccumulator_max,
                 float& DGAccumulator_min, struct CUDA_Context* ctx);
void Sanity_allNodes_cuda(float& DGAccumulator_sum, float& DGAccumulator_max,
                          float& DGAccumulator_min, struct CUDA_Context* ctx);
void Sanity_masterNodes_cuda(float& DGAccumulator_sum, float& DGAccumulator_max,
                             float& DGAccumulator_min,
                             struct CUDA_Context* ctx);
void Sanity_nodesWithEdges_cuda(float& DGAccumulator_sum,
                                float& DGAccumulator_max,
                                float& DGAccumulator_min,
                                struct CUDA_Context* ctx);
