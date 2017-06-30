
struct Broadcast_comp_current {
  static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) return get_node_comp_current_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.comp_current;
  }
  static bool extract_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t *s, DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_node_comp_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static bool extract_batch(unsigned from_id, unsigned int *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_node_comp_current_cuda(cuda_ctx, from_id, y); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static void setVal (uint32_t node_id, struct NodeData & node, unsigned int y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) set_node_comp_current_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
      node.comp_current = y;
  }
  static bool setVal_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t s, DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_set_mirror_node_comp_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  typedef unsigned int ValTy;
};

struct Reduce_set_comp_current {
  static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) return get_node_comp_current_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.comp_current;
  }
  static bool extract_reset_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t *s, DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_mirror_node_comp_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_mirror_node_comp_current_cuda(cuda_ctx, from_id, y); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static bool reduce (uint32_t node_id, struct NodeData & node, unsigned int y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) set_node_comp_current_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
      { Galois::set(node.comp_current, y); }
    return true;
  }
  static bool reduce_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t s, DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_set_node_comp_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static void reset (uint32_t node_id, struct NodeData & node ) {
  }
  typedef unsigned int ValTy;
};

struct Reduce_min_comp_current {
  static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) return get_node_comp_current_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.comp_current;
  }
  static bool extract_reset_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t *s, DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_mirror_node_comp_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_mirror_node_comp_current_cuda(cuda_ctx, from_id, y); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static bool reduce (uint32_t node_id, struct NodeData & node, unsigned int y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) return min_node_comp_current_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
      { return y < Galois::min(node.comp_current, y); }
    return false;
  }
  static bool reduce_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t s, DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_min_node_comp_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static void reset (uint32_t node_id, struct NodeData & node ) {
  }
  typedef unsigned int ValTy;
};
