
struct Reduce_add_nout {
  static unsigned int extract(uint32_t node_id, const struct PR_NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) return get_node_nout_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.nout;
  }
  static bool extract_reset_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t *s, DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_reset_node_nout_cuda(cuda_ctx, from_id, b, o, y, s, data_mode, 0); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_reset_node_nout_cuda(cuda_ctx, from_id, y, 0); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static bool reduce (uint32_t node_id, struct PR_NodeData & node, unsigned int y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) add_node_nout_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
      { Galois::add(node.nout, y); }
    return true;
  }
  static bool reduce_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t s, DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_add_node_nout_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static void reset (uint32_t node_id, struct PR_NodeData & node ) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) set_node_nout_cuda(cuda_ctx, node_id, 0);
    else if (personality == CPU)
  #endif
      { node.nout = 0; }
  }
  typedef unsigned int ValTy;
};

struct Broadcast_nout {
  static unsigned int extract(uint32_t node_id, const struct PR_NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) return get_node_nout_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.nout;
  }
  static bool extract_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t *s, DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_node_nout_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static bool extract_batch(unsigned from_id, unsigned int *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_node_nout_cuda(cuda_ctx, from_id, y); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static void setVal (uint32_t node_id, struct PR_NodeData & node, unsigned int y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) set_node_nout_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
      node.nout = y;
  }
  static bool setVal_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, unsigned int *y, size_t s, DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_set_mirror_node_nout_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  typedef unsigned int ValTy;
};

struct Reduce_add_residual {
  static float extract(uint32_t node_id, const struct PR_NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) return get_node_residual_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.residual;
  }
  static bool extract_reset_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, float *y, size_t *s, DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_reset_node_residual_cuda(cuda_ctx, from_id, b, o, y, s, data_mode, 0); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static bool extract_reset_batch(unsigned from_id, float *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_reset_node_residual_cuda(cuda_ctx, from_id, y, 0); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static bool reduce (uint32_t node_id, struct PR_NodeData & node, float y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) add_node_residual_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
      { Galois::add(node.residual, y); }
    return true;
  }
  static bool reduce_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, float *y, size_t s, DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_add_node_residual_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static void reset (uint32_t node_id, struct PR_NodeData & node ) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) set_node_residual_cuda(cuda_ctx, node_id, 0);
    else if (personality == CPU)
  #endif
      { node.residual = 0; }
  }
  typedef float ValTy;
};

struct Broadcast_residual {
  static float extract(uint32_t node_id, const struct PR_NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) return get_node_residual_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.residual;
  }
  static bool extract_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, float *y, size_t *s, DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_node_residual_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static bool extract_batch(unsigned from_id, float *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_get_node_residual_cuda(cuda_ctx, from_id, y); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  static void setVal (uint32_t node_id, struct PR_NodeData & node, float y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) set_node_residual_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
      node.residual = y;
  }
  static bool setVal_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, float *y, size_t s, DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { batch_set_mirror_node_residual_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }
    assert (personality == CPU);
  #endif
    return false;
  }
  typedef float ValTy;
};

struct Bitset_residual {
  static bool is_valid() {
    return true;
  }
  static Galois::DynamicBitSet& get() {
    if (personality == GPU_CUDA) get_bitset_residual_cuda(cuda_ctx, (unsigned long long int *)bitset_residual.get_vec().data());
    return bitset_residual;
  }
  // inclusive range
  static void reset_range(size_t begin, size_t end) {
    if (personality == GPU_CUDA) {
      bitset_residual_reset_cuda(cuda_ctx, begin, end);
    } else {
      assert (personality == CPU);
      bitset_residual.reset(begin, end);
    }
  }
};

struct Bitset_nout {
  static bool is_valid() {
    return true;
  }
  static Galois::DynamicBitSet& get() {
    if (personality == GPU_CUDA) get_bitset_nout_cuda(cuda_ctx, (unsigned long long int *)bitset_nout.get_vec().data());
    return bitset_nout;
  }
  // inclusive range
  static void reset_range(size_t begin, size_t end) {
    if (personality == GPU_CUDA) {
      bitset_nout_reset_cuda(cuda_ctx, begin, end);
    } else {
      assert (personality == CPU);
      bitset_nout.reset(begin, end);
    }
  }
};
