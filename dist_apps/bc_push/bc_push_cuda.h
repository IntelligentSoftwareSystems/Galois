#pragma once

#include "galois/runtime/DataCommMode.h"
#include "galois/cuda/HostDecls.h"

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
                                               unsigned from_id, float* v);
void batch_get_node_betweeness_centrality_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, float* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_mirror_node_betweeness_centrality_cuda(struct CUDA_Context* ctx,
                                                      unsigned from_id,
                                                      float* v);
void batch_get_mirror_node_betweeness_centrality_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, float* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_reset_node_betweeness_centrality_cuda(struct CUDA_Context* ctx,
                                                     unsigned from_id, float* v,
                                                     float i);
void batch_get_reset_node_betweeness_centrality_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, float* v, size_t* v_size, DataCommMode* data_mode,
    float i);
void batch_set_mirror_node_betweeness_centrality_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, float* v, size_t v_size, DataCommMode data_mode);
void batch_set_node_betweeness_centrality_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, float* v, size_t v_size, DataCommMode data_mode);
void batch_add_node_betweeness_centrality_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, float* v, size_t v_size, DataCommMode data_mode);
void batch_min_node_betweeness_centrality_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, float* v, size_t v_size, DataCommMode data_mode);
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
                                        unsigned from_id, uint32_t* v);
void batch_get_node_current_length_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint64_t* bitset_comm,
                                        unsigned int* offsets, uint32_t* v,
                                        size_t* v_size,
                                        DataCommMode* data_mode);
void batch_get_mirror_node_current_length_cuda(struct CUDA_Context* ctx,
                                               unsigned from_id, uint32_t* v);
void batch_get_mirror_node_current_length_cuda(struct CUDA_Context* ctx,
                                               unsigned from_id,
                                               uint64_t* bitset_comm,
                                               unsigned int* offsets,
                                               uint32_t* v, size_t* v_size,
                                               DataCommMode* data_mode);
void batch_get_reset_node_current_length_cuda(struct CUDA_Context* ctx,
                                              unsigned from_id, uint32_t* v,
                                              uint32_t i);
void batch_get_reset_node_current_length_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint32_t* v, size_t* v_size, DataCommMode* data_mode,
    uint32_t i);
void batch_set_mirror_node_current_length_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint32_t* v, size_t v_size, DataCommMode data_mode);
void batch_set_node_current_length_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint64_t* bitset_comm,
                                        unsigned int* offsets, uint32_t* v,
                                        size_t v_size, DataCommMode data_mode);
void batch_add_node_current_length_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint64_t* bitset_comm,
                                        unsigned int* offsets, uint32_t* v,
                                        size_t v_size, DataCommMode data_mode);
void batch_min_node_current_length_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint64_t* bitset_comm,
                                        unsigned int* offsets, uint32_t* v,
                                        size_t v_size, DataCommMode data_mode);
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
                                    float* v);
void batch_get_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint64_t* bitset_comm,
                                    unsigned int* offsets, float* v,
                                    size_t* v_size, DataCommMode* data_mode);
void batch_get_mirror_node_dependency_cuda(struct CUDA_Context* ctx,
                                           unsigned from_id, float* v);
void batch_get_mirror_node_dependency_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, float* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_reset_node_dependency_cuda(struct CUDA_Context* ctx,
                                          unsigned from_id, float* v, float i);
void batch_get_reset_node_dependency_cuda(struct CUDA_Context* ctx,
                                          unsigned from_id,
                                          uint64_t* bitset_comm,
                                          unsigned int* offsets, float* v,
                                          size_t* v_size,
                                          DataCommMode* data_mode, float i);
void batch_set_mirror_node_dependency_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, float* v, size_t v_size, DataCommMode data_mode);
void batch_set_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint64_t* bitset_comm,
                                    unsigned int* offsets, float* v,
                                    size_t v_size, DataCommMode data_mode);
void batch_add_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint64_t* bitset_comm,
                                    unsigned int* offsets, float* v,
                                    size_t v_size, DataCommMode data_mode);
void batch_min_node_dependency_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint64_t* bitset_comm,
                                    unsigned int* offsets, float* v,
                                    size_t v_size, DataCommMode data_mode);
void batch_reset_node_dependency_cuda(struct CUDA_Context* ctx, size_t begin,
                                      size_t end, float v);

void get_bitset_num_predecessors_cuda(struct CUDA_Context* ctx,
                                      uint64_t* bitset_compute);
void bitset_num_predecessors_reset_cuda(struct CUDA_Context* ctx);
void bitset_num_predecessors_reset_cuda(struct CUDA_Context* ctx, size_t begin,
                                        size_t end);
uint32_t get_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned LID,
                                    uint32_t v);
void add_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned LID,
                                    uint32_t v);
bool min_node_num_predecessors_cuda(struct CUDA_Context* ctx, unsigned LID,
                                    uint32_t v);
void batch_get_node_num_predecessors_cuda(struct CUDA_Context* ctx,
                                          unsigned from_id, uint32_t* v);
void batch_get_node_num_predecessors_cuda(struct CUDA_Context* ctx,
                                          unsigned from_id,
                                          uint64_t* bitset_comm,
                                          unsigned int* offsets, uint32_t* v,
                                          size_t* v_size,
                                          DataCommMode* data_mode);
void batch_get_mirror_node_num_predecessors_cuda(struct CUDA_Context* ctx,
                                                 unsigned from_id, uint32_t* v);
void batch_get_mirror_node_num_predecessors_cuda(struct CUDA_Context* ctx,
                                                 unsigned from_id,
                                                 uint64_t* bitset_comm,
                                                 unsigned int* offsets,
                                                 uint32_t* v, size_t* v_size,
                                                 DataCommMode* data_mode);
void batch_get_reset_node_num_predecessors_cuda(struct CUDA_Context* ctx,
                                                unsigned from_id, uint32_t* v,
                                                uint32_t i);
void batch_get_reset_node_num_predecessors_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint32_t* v, size_t* v_size, DataCommMode* data_mode,
    uint32_t i);
void batch_set_mirror_node_num_predecessors_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint32_t* v, size_t v_size, DataCommMode data_mode);
void batch_set_node_num_predecessors_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint32_t* v, size_t v_size, DataCommMode data_mode);
void batch_add_node_num_predecessors_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint32_t* v, size_t v_size, DataCommMode data_mode);
void batch_min_node_num_predecessors_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint32_t* v, size_t v_size, DataCommMode data_mode);
void batch_reset_node_num_predecessors_cuda(struct CUDA_Context* ctx,
                                            size_t begin, size_t end,
                                            uint32_t v);

void get_bitset_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                        uint64_t* bitset_compute);
void bitset_num_shortest_paths_reset_cuda(struct CUDA_Context* ctx);
void bitset_num_shortest_paths_reset_cuda(struct CUDA_Context* ctx,
                                          size_t begin, size_t end);
uint64_t get_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                          unsigned LID);
void set_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned LID,
                                      uint64_t v);
void add_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned LID,
                                      uint64_t v);
bool min_node_num_shortest_paths_cuda(struct CUDA_Context* ctx, unsigned LID,
                                      uint64_t v);
void batch_get_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                            unsigned from_id, uint64_t* v);
void batch_get_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                            unsigned from_id,
                                            uint64_t* bitset_comm,
                                            unsigned int* offsets, uint64_t* v,
                                            size_t* v_size,
                                            DataCommMode* data_mode);
void batch_get_mirror_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                                   unsigned from_id,
                                                   uint64_t* v);
void batch_get_mirror_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                                   unsigned from_id,
                                                   uint64_t* bitset_comm,
                                                   unsigned int* offsets,
                                                   uint64_t* v, size_t* v_size,
                                                   DataCommMode* data_mode);
void batch_get_reset_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                                  unsigned from_id, uint64_t* v,
                                                  uint64_t i);
void batch_get_reset_node_num_shortest_paths_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint64_t* v, size_t* v_size, DataCommMode* data_mode,
    uint64_t i);
void batch_set_mirror_node_num_shortest_paths_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint64_t* v, size_t v_size, DataCommMode data_mode);
void batch_set_node_num_shortest_paths_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint64_t* v, size_t v_size, DataCommMode data_mode);
void batch_add_node_num_shortest_paths_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint64_t* v, size_t v_size, DataCommMode data_mode);
void batch_min_node_num_shortest_paths_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint64_t* v, size_t v_size, DataCommMode data_mode);
void batch_reset_node_num_shortest_paths_cuda(struct CUDA_Context* ctx,
                                              size_t begin, size_t end,
                                              uint64_t v);

void get_bitset_num_successors_cuda(struct CUDA_Context* ctx,
                                    uint64_t* bitset_compute);
void bitset_num_successors_reset_cuda(struct CUDA_Context* ctx);
void bitset_num_successors_reset_cuda(struct CUDA_Context* ctx, size_t begin,
                                      size_t end);
uint32_t get_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned LID,
                                  uint32_t v);
void add_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned LID,
                                  uint32_t v);
bool min_node_num_successors_cuda(struct CUDA_Context* ctx, unsigned LID,
                                  uint32_t v);
void batch_get_node_num_successors_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint32_t* v);
void batch_get_node_num_successors_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint64_t* bitset_comm,
                                        unsigned int* offsets, uint32_t* v,
                                        size_t* v_size,
                                        DataCommMode* data_mode);
void batch_get_mirror_node_num_successors_cuda(struct CUDA_Context* ctx,
                                               unsigned from_id, uint32_t* v);
void batch_get_mirror_node_num_successors_cuda(struct CUDA_Context* ctx,
                                               unsigned from_id,
                                               uint64_t* bitset_comm,
                                               unsigned int* offsets,
                                               uint32_t* v, size_t* v_size,
                                               DataCommMode* data_mode);
void batch_get_reset_node_num_successors_cuda(struct CUDA_Context* ctx,
                                              unsigned from_id, uint32_t* v,
                                              uint32_t i);
void batch_get_reset_node_num_successors_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint32_t* v, size_t* v_size, DataCommMode* data_mode,
    uint32_t i);
void batch_set_mirror_node_num_successors_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint32_t* v, size_t v_size, DataCommMode data_mode);
void batch_set_node_num_successors_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint64_t* bitset_comm,
                                        unsigned int* offsets, uint32_t* v,
                                        size_t v_size, DataCommMode data_mode);
void batch_add_node_num_successors_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint64_t* bitset_comm,
                                        unsigned int* offsets, uint32_t* v,
                                        size_t v_size, DataCommMode data_mode);
void batch_min_node_num_successors_cuda(struct CUDA_Context* ctx,
                                        unsigned from_id, uint64_t* bitset_comm,
                                        unsigned int* offsets, uint32_t* v,
                                        size_t v_size, DataCommMode data_mode);
void batch_reset_node_num_successors_cuda(struct CUDA_Context* ctx,
                                          size_t begin, size_t end, uint32_t v);

void get_bitset_old_length_cuda(struct CUDA_Context* ctx,
                                uint64_t* bitset_compute);
void bitset_old_length_reset_cuda(struct CUDA_Context* ctx);
void bitset_old_length_reset_cuda(struct CUDA_Context* ctx, size_t begin,
                                  size_t end);
uint32_t get_node_old_length_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_old_length_cuda(struct CUDA_Context* ctx, unsigned LID,
                              uint32_t v);
void add_node_old_length_cuda(struct CUDA_Context* ctx, unsigned LID,
                              uint32_t v);
bool min_node_old_length_cuda(struct CUDA_Context* ctx, unsigned LID,
                              uint32_t v);
void batch_get_node_old_length_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint32_t* v);
void batch_get_node_old_length_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint64_t* bitset_comm,
                                    unsigned int* offsets, uint32_t* v,
                                    size_t* v_size, DataCommMode* data_mode);
void batch_get_mirror_node_old_length_cuda(struct CUDA_Context* ctx,
                                           unsigned from_id, uint32_t* v);
void batch_get_mirror_node_old_length_cuda(struct CUDA_Context* ctx,
                                           unsigned from_id,
                                           uint64_t* bitset_comm,
                                           unsigned int* offsets, uint32_t* v,
                                           size_t* v_size,
                                           DataCommMode* data_mode);
void batch_get_reset_node_old_length_cuda(struct CUDA_Context* ctx,
                                          unsigned from_id, uint32_t* v,
                                          uint32_t i);
void batch_get_reset_node_old_length_cuda(struct CUDA_Context* ctx,
                                          unsigned from_id,
                                          uint64_t* bitset_comm,
                                          unsigned int* offsets, uint32_t* v,
                                          size_t* v_size,
                                          DataCommMode* data_mode, uint32_t i);
void batch_set_mirror_node_old_length_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint32_t* v, size_t v_size, DataCommMode data_mode);
void batch_set_node_old_length_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint64_t* bitset_comm,
                                    unsigned int* offsets, uint32_t* v,
                                    size_t v_size, DataCommMode data_mode);
void batch_add_node_old_length_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint64_t* bitset_comm,
                                    unsigned int* offsets, uint32_t* v,
                                    size_t v_size, DataCommMode data_mode);
void batch_min_node_old_length_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                    uint64_t* bitset_comm,
                                    unsigned int* offsets, uint32_t* v,
                                    size_t v_size, DataCommMode data_mode);
void batch_reset_node_old_length_cuda(struct CUDA_Context* ctx, size_t begin,
                                      size_t end, uint32_t v);

void get_bitset_propagation_flag_cuda(struct CUDA_Context* ctx,
                                      uint64_t* bitset_compute);
void bitset_propagation_flag_reset_cuda(struct CUDA_Context* ctx);
void bitset_propagation_flag_reset_cuda(struct CUDA_Context* ctx, size_t begin,
                                        size_t end);
uint8_t get_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned LID,
                                    uint8_t v);
void add_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned LID,
                                    uint8_t v);
bool min_node_propagation_flag_cuda(struct CUDA_Context* ctx, unsigned LID,
                                    uint8_t v);
void batch_get_node_propagation_flag_cuda(struct CUDA_Context* ctx,
                                          unsigned from_id, uint8_t* v);
void batch_get_node_propagation_flag_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint8_t* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_mirror_node_propagation_flag_cuda(struct CUDA_Context* ctx,
                                                 unsigned from_id, uint8_t* v);
void batch_get_mirror_node_propagation_flag_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint8_t* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_reset_node_propagation_flag_cuda(struct CUDA_Context* ctx,
                                                unsigned from_id, uint8_t* v,
                                                uint8_t i);
void batch_get_reset_node_propagation_flag_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint8_t* v, size_t* v_size, DataCommMode* data_mode,
    uint8_t i);
void batch_set_mirror_node_propagation_flag_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint8_t* v, size_t v_size, DataCommMode data_mode);
void batch_set_node_propagation_flag_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint8_t* v, size_t v_size, DataCommMode data_mode);
void batch_add_node_propagation_flag_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint8_t* v, size_t v_size, DataCommMode data_mode);
void batch_min_node_propagation_flag_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, uint8_t* v, size_t v_size, DataCommMode data_mode);
void batch_reset_node_propagation_flag_cuda(struct CUDA_Context* ctx,
                                            size_t begin, size_t end,
                                            uint8_t v);

void get_bitset_to_add_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute);
void bitset_to_add_reset_cuda(struct CUDA_Context* ctx);
void bitset_to_add_reset_cuda(struct CUDA_Context* ctx, size_t begin,
                              size_t end);
uint64_t get_node_to_add_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_to_add_cuda(struct CUDA_Context* ctx, unsigned LID, uint64_t v);
void add_node_to_add_cuda(struct CUDA_Context* ctx, unsigned LID, uint64_t v);
bool min_node_to_add_cuda(struct CUDA_Context* ctx, unsigned LID, uint64_t v);
void batch_get_node_to_add_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                uint64_t* v);
void batch_get_node_to_add_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                uint64_t* bitset_comm, unsigned int* offsets,
                                uint64_t* v, size_t* v_size,
                                DataCommMode* data_mode);
void batch_get_mirror_node_to_add_cuda(struct CUDA_Context* ctx,
                                       unsigned from_id, uint64_t* v);
void batch_get_mirror_node_to_add_cuda(struct CUDA_Context* ctx,
                                       unsigned from_id, uint64_t* bitset_comm,
                                       unsigned int* offsets, uint64_t* v,
                                       size_t* v_size, DataCommMode* data_mode);
void batch_get_reset_node_to_add_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint64_t* v,
                                      uint64_t i);
void batch_get_reset_node_to_add_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint64_t* bitset_comm,
                                      unsigned int* offsets, uint64_t* v,
                                      size_t* v_size, DataCommMode* data_mode,
                                      uint64_t i);
void batch_set_mirror_node_to_add_cuda(struct CUDA_Context* ctx,
                                       unsigned from_id, uint64_t* bitset_comm,
                                       unsigned int* offsets, uint64_t* v,
                                       size_t v_size, DataCommMode data_mode);
void batch_set_node_to_add_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                uint64_t* bitset_comm, unsigned int* offsets,
                                uint64_t* v, size_t v_size,
                                DataCommMode data_mode);
void batch_add_node_to_add_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                uint64_t* bitset_comm, unsigned int* offsets,
                                uint64_t* v, size_t v_size,
                                DataCommMode data_mode);
void batch_min_node_to_add_cuda(struct CUDA_Context* ctx, unsigned from_id,
                                uint64_t* bitset_comm, unsigned int* offsets,
                                uint64_t* v, size_t v_size,
                                DataCommMode data_mode);
void batch_reset_node_to_add_cuda(struct CUDA_Context* ctx, size_t begin,
                                  size_t end, uint64_t v);

void get_bitset_to_add_float_cuda(struct CUDA_Context* ctx,
                                  uint64_t* bitset_compute);
void bitset_to_add_float_reset_cuda(struct CUDA_Context* ctx);
void bitset_to_add_float_reset_cuda(struct CUDA_Context* ctx, size_t begin,
                                    size_t end);
float get_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned LID,
                                float v);
void add_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned LID,
                                float v);
bool min_node_to_add_float_cuda(struct CUDA_Context* ctx, unsigned LID,
                                float v);
void batch_get_node_to_add_float_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, float* v);
void batch_get_node_to_add_float_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint64_t* bitset_comm,
                                      unsigned int* offsets, float* v,
                                      size_t* v_size, DataCommMode* data_mode);
void batch_get_mirror_node_to_add_float_cuda(struct CUDA_Context* ctx,
                                             unsigned from_id, float* v);
void batch_get_mirror_node_to_add_float_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, float* v, size_t* v_size, DataCommMode* data_mode);
void batch_get_reset_node_to_add_float_cuda(struct CUDA_Context* ctx,
                                            unsigned from_id, float* v,
                                            float i);
void batch_get_reset_node_to_add_float_cuda(struct CUDA_Context* ctx,
                                            unsigned from_id,
                                            uint64_t* bitset_comm,
                                            unsigned int* offsets, float* v,
                                            size_t* v_size,
                                            DataCommMode* data_mode, float i);
void batch_set_mirror_node_to_add_float_cuda(
    struct CUDA_Context* ctx, unsigned from_id, uint64_t* bitset_comm,
    unsigned int* offsets, float* v, size_t v_size, DataCommMode data_mode);
void batch_set_node_to_add_float_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint64_t* bitset_comm,
                                      unsigned int* offsets, float* v,
                                      size_t v_size, DataCommMode data_mode);
void batch_add_node_to_add_float_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint64_t* bitset_comm,
                                      unsigned int* offsets, float* v,
                                      size_t v_size, DataCommMode data_mode);
void batch_min_node_to_add_float_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint64_t* bitset_comm,
                                      unsigned int* offsets, float* v,
                                      size_t v_size, DataCommMode data_mode);
void batch_reset_node_to_add_float_cuda(struct CUDA_Context* ctx, size_t begin,
                                        size_t end, float v);

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

void BC_cuda(unsigned int __begin, unsigned int __end,
             struct CUDA_Context* ctx);
void BC_allNodes_cuda(struct CUDA_Context* ctx);
void BC_masterNodes_cuda(struct CUDA_Context* ctx);
void BC_nodesWithEdges_cuda(struct CUDA_Context* ctx);
void DependencyPropChanges_cuda(unsigned int __begin, unsigned int __end,
                                const uint32_t& local_infinity,
                                struct CUDA_Context* ctx);
void DependencyPropChanges_allNodes_cuda(const uint32_t& local_infinity,
                                         struct CUDA_Context* ctx);
void DependencyPropChanges_masterNodes_cuda(const uint32_t& local_infinity,
                                            struct CUDA_Context* ctx);
void DependencyPropChanges_nodesWithEdges_cuda(const uint32_t& local_infinity,
                                               struct CUDA_Context* ctx);
void DependencyPropagation_cuda(unsigned int __begin, unsigned int __end,
                                uint32_t& DGAccumulator_accum,
                                const uint32_t& local_infinity,
                                const uint64_t& local_current_src_node,
                                struct CUDA_Context* ctx);
void DependencyPropagation_allNodes_cuda(uint32_t& DGAccumulator_accum,
                                         const uint32_t& local_infinity,
                                         const uint64_t& local_current_src_node,
                                         struct CUDA_Context* ctx);
void DependencyPropagation_masterNodes_cuda(
    uint32_t& DGAccumulator_accum, const uint32_t& local_infinity,
    const uint64_t& local_current_src_node, struct CUDA_Context* ctx);
void DependencyPropagation_nodesWithEdges_cuda(
    uint32_t& DGAccumulator_accum, const uint32_t& local_infinity,
    const uint64_t& local_current_src_node, struct CUDA_Context* ctx);
void FirstIterationSSSP_cuda(unsigned int __begin, unsigned int __end,
                             struct CUDA_Context* ctx);
void FirstIterationSSSP_allNodes_cuda(struct CUDA_Context* ctx);
void FirstIterationSSSP_masterNodes_cuda(struct CUDA_Context* ctx);
void FirstIterationSSSP_nodesWithEdges_cuda(struct CUDA_Context* ctx);
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
void NumShortestPaths_cuda(unsigned int __begin, unsigned int __end,
                           uint32_t& DGAccumulator_accum,
                           const uint32_t& local_infinity,
                           const uint64_t local_current_src_node,
                           struct CUDA_Context* ctx);
void NumShortestPathsChanges_cuda(unsigned int __begin, unsigned int __end,
                                  const uint32_t& local_infinity,
                                  struct CUDA_Context* ctx);
void NumShortestPathsChanges_allNodes_cuda(const uint32_t& local_infinity,
                                           struct CUDA_Context* ctx);
void NumShortestPathsChanges_masterNodes_cuda(const uint32_t& local_infinity,
                                              struct CUDA_Context* ctx);
void NumShortestPathsChanges_nodesWithEdges_cuda(const uint32_t& local_infinity,
                                                 struct CUDA_Context* ctx);
void NumShortestPaths_allNodes_cuda(uint32_t& DGAccumulator_accum,
                                    const uint32_t& local_infinity,
                                    const uint64_t local_current_src_node,
                                    struct CUDA_Context* ctx);
void NumShortestPaths_masterNodes_cuda(uint32_t& DGAccumulator_accum,
                                       const uint32_t& local_infinity,
                                       const uint64_t local_current_src_node,
                                       struct CUDA_Context* ctx);
void NumShortestPaths_nodesWithEdges_cuda(uint32_t& DGAccumulator_accum,
                                          const uint32_t& local_infinity,
                                          const uint64_t local_current_src_node,
                                          struct CUDA_Context* ctx);
void PredAndSucc_cuda(unsigned int __begin, unsigned int __end,
                      const uint32_t& local_infinity, struct CUDA_Context* ctx);
void PredAndSucc_allNodes_cuda(const uint32_t& local_infinity,
                               struct CUDA_Context* ctx);
void PredAndSucc_masterNodes_cuda(const uint32_t& local_infinity,
                                  struct CUDA_Context* ctx);
void PredAndSucc_nodesWithEdges_cuda(const uint32_t& local_infinity,
                                     struct CUDA_Context* ctx);
void PropagationFlagUpdate_cuda(unsigned int __begin, unsigned int __end,
                                const uint32_t& local_infinity,
                                struct CUDA_Context* ctx);
void PropagationFlagUpdate_allNodes_cuda(const uint32_t& local_infinity,
                                         struct CUDA_Context* ctx);
void PropagationFlagUpdate_masterNodes_cuda(const uint32_t& local_infinity,
                                            struct CUDA_Context* ctx);
void PropagationFlagUpdate_nodesWithEdges_cuda(const uint32_t& local_infinity,
                                               struct CUDA_Context* ctx);
void SSSP_cuda(unsigned int __begin, unsigned int __end,
               uint32_t& DGAccumulator_accum, struct CUDA_Context* ctx);
void SSSP_allNodes_cuda(uint32_t& DGAccumulator_accum,
                        struct CUDA_Context* ctx);
void SSSP_masterNodes_cuda(uint32_t& DGAccumulator_accum,
                           struct CUDA_Context* ctx);
void SSSP_nodesWithEdges_cuda(uint32_t& DGAccumulator_accum,
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
