
struct Broadcast_dist_current {
  static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) return get_node_dist_current_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.dist_current;
  }
  static bool extract_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t *s, DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static bool extract_batch(unsigned from_id, unsigned int *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_node_dist_current_cuda(cuda_ctx, from_id, y); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static void setVal (uint32_t node_id, struct NodeData & node, unsigned int y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) set_node_dist_current_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
      node.dist_current = y;
  }
  static bool setVal_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t s, DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_set_mirror_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  typedef unsigned int ValTy;
};

struct Reduce_set_dist_current {
  static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) return get_node_dist_current_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.dist_current;
  }
  static bool extract_reset_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t *s, DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_mirror_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_mirror_node_dist_current_cuda(cuda_ctx, from_id, y); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static bool reduce (uint32_t node_id, struct NodeData & node, unsigned int y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) set_node_dist_current_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
      { Galois::set(node.dist_current, y); }
    return true;
  }
  static bool reduce_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t s, DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_set_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static void reset (uint32_t node_id, struct NodeData & node ) {
  }
  typedef unsigned int ValTy;
};

struct Reduce_min_dist_current {
  static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) return get_node_dist_current_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.dist_current;
  }
  static bool extract_reset_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t *s, DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_mirror_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_mirror_node_dist_current_cuda(cuda_ctx, from_id, y); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static bool reduce (uint32_t node_id, struct NodeData & node, unsigned int y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) return min_node_dist_current_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
      { return y < Galois::min(node.dist_current, y); }
    return false;
  }
  static bool reduce_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t s, DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_min_node_dist_current_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static void reset (uint32_t node_id, struct NodeData & node ) {
  }
  typedef unsigned int ValTy;
};

struct Bitset_dist_current {
  static bool is_valid() {
    return true;
  }
  static Galois::DynamicBitSet& get() {
    if (personality == GPU_CUDA) get_bitset_dist_current_cuda(cuda_ctx, (unsigned long long int *)bitset_dist_current.get_vec().data());
    return bitset_dist_current;
  }
  // inclusive range
  static void reset_range(size_t begin, size_t end) {
    if (personality == GPU_CUDA) {
      bitset_dist_current_reset_cuda(cuda_ctx, begin, end);
    } else {
      assert (personality == CPU);
      bitset_dist_current.reset(begin, end);
    }
  }
};
