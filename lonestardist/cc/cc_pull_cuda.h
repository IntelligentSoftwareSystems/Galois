#pragma once

#include "galois/runtime/DataCommMode.h"
#include "galois/cuda/HostDecls.h"

void get_bitset_comp_current_cuda(struct CUDA_Context* ctx,
                                  uint64_t* bitset_compute);
void bitset_comp_current_reset_cuda(struct CUDA_Context* ctx);
void bitset_comp_current_reset_cuda(struct CUDA_Context* ctx, size_t begin,
                                    size_t end);
uint32_t get_node_comp_current_cuda(struct CUDA_Context* ctx, unsigned LID);
void set_node_comp_current_cuda(struct CUDA_Context* ctx, unsigned LID,
                                uint32_t v);
void add_node_comp_current_cuda(struct CUDA_Context* ctx, unsigned LID,
                                uint32_t v);
bool min_node_comp_current_cuda(struct CUDA_Context* ctx, unsigned LID,
                                uint32_t v);
void batch_get_node_comp_current_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint8_t* v);
void batch_get_node_comp_current_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint8_t* v,
                                      size_t* v_size, DataCommMode* data_mode);
void batch_get_mirror_node_comp_current_cuda(struct CUDA_Context* ctx,
                                             unsigned from_id, uint8_t* v);
void batch_get_mirror_node_comp_current_cuda(struct CUDA_Context* ctx,
                                             unsigned from_id, uint8_t* v,
                                             size_t* v_size,
                                             DataCommMode* data_mode);
void batch_get_reset_node_comp_current_cuda(struct CUDA_Context* ctx,
                                            unsigned from_id, uint8_t* v,
                                            uint32_t i);
void batch_get_reset_node_comp_current_cuda(struct CUDA_Context* ctx,
                                            unsigned from_id, uint8_t* v,
                                            size_t* v_size,
                                            DataCommMode* data_mode,
                                            uint32_t i);
void batch_set_mirror_node_comp_current_cuda(struct CUDA_Context* ctx,
                                             unsigned from_id, uint8_t* v,
                                             DataCommMode data_mode);
void batch_set_node_comp_current_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint8_t* v,
                                      DataCommMode data_mode);
void batch_add_mirror_node_comp_current_cuda(struct CUDA_Context* ctx,
                                             unsigned from_id, uint8_t* v,
                                             DataCommMode data_mode);
void batch_add_node_comp_current_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint8_t* v,
                                      DataCommMode data_mode);
void batch_min_mirror_node_comp_current_cuda(struct CUDA_Context* ctx,
                                             unsigned from_id, uint8_t* v,
                                             DataCommMode data_mode);
void batch_min_node_comp_current_cuda(struct CUDA_Context* ctx,
                                      unsigned from_id, uint8_t* v,
                                      DataCommMode data_mode);
void batch_reset_node_comp_current_cuda(struct CUDA_Context* ctx, size_t begin,
                                        size_t end, uint32_t v);

void ConnectedComp_cuda(unsigned int __begin, unsigned int __end,
                        unsigned int& active_vertices,
                        struct CUDA_Context* ctx);
void ConnectedCompSanityCheck_cuda(unsigned int __begin, unsigned int __end,
                                   uint64_t& active_vertices,
                                   struct CUDA_Context* ctx);
void ConnectedCompSanityCheck_allNodes_cuda(uint64_t& active_vertices,
                                            struct CUDA_Context* ctx);
void ConnectedCompSanityCheck_masterNodes_cuda(uint64_t& active_vertices,
                                               struct CUDA_Context* ctx);
void ConnectedCompSanityCheck_nodesWithEdges_cuda(uint64_t& active_vertices,
                                                  struct CUDA_Context* ctx);
void ConnectedComp_allNodes_cuda(unsigned int& active_vertices,
                                 struct CUDA_Context* ctx);
void ConnectedComp_masterNodes_cuda(unsigned int& active_vertices,
                                    struct CUDA_Context* ctx);
void ConnectedComp_nodesWithEdges_cuda(unsigned int& active_vertices,
                                       struct CUDA_Context* ctx);
void InitializeGraph_cuda(unsigned int __begin, unsigned int __end,
                          struct CUDA_Context* ctx);
void InitializeGraph_allNodes_cuda(struct CUDA_Context* ctx);
void InitializeGraph_masterNodes_cuda(struct CUDA_Context* ctx);
void InitializeGraph_nodesWithEdges_cuda(struct CUDA_Context* ctx);
