////////////////////////////////////////////////////////////////////////////////
// current_degree
////////////////////////////////////////////////////////////////////////////////
struct Reduce_current_degree {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_current_degree_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.current_degree;
  }

  static bool extract_reset_batch(unsigned from_id, 
                                  unsigned long long int *b, 
                                  unsigned int *o, 
                                  unsigned int *y, 
                                  size_t *s, 
                                  DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { 
      batch_get_mirror_node_current_degree_cuda(cuda_ctx, from_id, b, o, y, s,
                                               data_mode); 
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_mirror_node_current_degree_cuda(cuda_ctx, from_id, y);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool reduce(uint32_t node_id, struct NodeData & node, ValTy y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      add_node_current_degree_cuda(cuda_ctx, node_id, y);
      return true;
    }
    //else if (personality == CPU)
    assert(personality == CPU);
  #endif
    { Galois::add(node.current_degree, y); return true;}
  }

  static bool reduce_batch(unsigned from_id, 
                           unsigned long long int *b, 
                           unsigned int *o, 
                           unsigned int *y, 
                           size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_add_node_current_degree_cuda(cuda_ctx, from_id, b, o, y, s, 
                                         data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static void reset (uint32_t node_id, struct NodeData & node) {
  }
};

struct ReduceSet_current_degree {
  typedef unsigned int ValTy;

  static ValTy extract(uint32_t node_id, const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_current_degree_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.current_degree;
  }

  static bool extract_reset_batch(unsigned from_id, 
                                  unsigned long long int *b, 
                                  unsigned int *o, 
                                  ValTy *y, 
                                  size_t *s, 
                                  DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { 
      batch_get_mirror_node_current_degree_cuda(cuda_ctx, from_id, b, o, y, s, data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool extract_reset_batch(unsigned from_id, ValTy *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_mirror_node_current_degree_cuda(cuda_ctx, from_id, y);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool reduce(uint32_t node_id, struct NodeData & node, 
                     ValTy y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      set_node_current_degree_cuda(cuda_ctx, node_id, y);
    }
    else if (personality == CPU)
  #endif
    Galois::set(node.current_degree, y);

    return true;
  }

  static bool reduce_batch(unsigned from_id, 
                           unsigned long long int *b, 
                           unsigned int *o, 
                           ValTy *y, 
                           size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_node_current_degree_cuda(cuda_ctx, from_id, b, o, y, s, 
                                         data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static void reset (uint32_t node_id, struct NodeData & node) {
    // no reset for reduce set
  }
};


struct Broadcast_current_degree {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_current_degree_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.current_degree;
  }

  static bool extract_batch(unsigned from_id,
                            unsigned long long int *b,
                            unsigned int *o,
                            unsigned int *y,
                            size_t *s, 
                            DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_node_current_degree_cuda(cuda_ctx, from_id, b, o, y, s,
                                         data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool extract_batch(unsigned from_id, unsigned int *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_node_current_degree_cuda(cuda_ctx, from_id, y);
      return true;
    }
    assert (personality == CPU);
  #endif

    return false;
  }

  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA)
      set_node_current_degree_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
    node.current_degree = y;
  }

  static bool setVal_batch(unsigned from_id, 
                           unsigned long long int *b, 
                           unsigned int *o, 
                           unsigned int *y, 
                           size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_mirror_current_degree_cuda(cuda_ctx, from_id, b, o, y, s, 
                                         data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif

    return false;
  }
};

struct Bitset_current_degree {
  static bool is_valid() {
    return true;
  }

  static Galois::DynamicBitSet& get() {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      get_bitset_current_degree_cuda(cuda_ctx, 
        (unsigned long long int *)bitset_current_degree.get_vec().data());
  #endif
    return bitset_current_degree;
  }

  // inclusive range
  static void reset_range(size_t begin, size_t end) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      bitset_current_degree_reset_cuda(cuda_ctx, begin, end);
    } else
  #endif
      {assert (personality == CPU);
      bitset_current_degree.reset(begin, end);}
  }
};

////////////////////////////////////////////////////////////////////////////////
// trim
////////////////////////////////////////////////////////////////////////////////

struct Reduce_trim {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_trim_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.trim;
  }

  static bool extract_reset_batch(unsigned from_id, 
                                  unsigned long long int *b, 
                                  unsigned int *o, 
                                  unsigned int *y, 
                                  size_t *s, 
                                  DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { 
      batch_get_reset_node_trim_cuda(cuda_ctx, from_id, b, o, y, s, 
                                     data_mode, 0);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_reset_node_trim_cuda(cuda_ctx, from_id, y, 0);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool reduce(uint32_t node_id, struct NodeData & node, ValTy y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      add_node_trim_cuda(cuda_ctx, node_id, y);
      return true;
    } 
    //else if (personality == CPU)
    assert(personality == CPU);
  #endif
    { return Galois::add(node.trim, y); return true; }
  }

  static bool reduce_batch(unsigned from_id, 
                           unsigned long long int *b, 
                           unsigned int *o, 
                           unsigned int *y, 
                           size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {

      batch_add_node_trim_cuda(cuda_ctx, from_id, b, o, y, s, 
                               data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static void reset (uint32_t node_id, struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      set_node_trim_cuda(cuda_ctx, node_id, 0);
    }
    else if (personality == CPU)
  #endif
    Galois::set(node.trim, (unsigned int)0);
  }

};

struct Broadcast_trim {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_trim_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.trim;
  }

  static bool extract_batch(unsigned from_id,
                            unsigned long long int *b,
                            unsigned int *o,
                            unsigned int *y,
                            size_t *s, 
                            DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_node_trim_cuda(cuda_ctx, from_id, b, o, y, s,
                               data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool extract_batch(unsigned from_id, unsigned int *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_node_trim_cuda(cuda_ctx, from_id, y);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA)
      set_node_trim_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
    Galois::set(node.trim, y);
  }

  static bool setVal_batch(unsigned from_id, 
                           unsigned long long int *b, 
                           unsigned int *o, 
                           unsigned int *y, 
                           size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_mirror_trim_cuda(cuda_ctx, from_id, b, o, y, s, 
                               data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }
};

struct Bitset_trim {
  static bool is_valid() {
    return true;
  }

  static Galois::DynamicBitSet& get() {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      get_bitset_trim_cuda(cuda_ctx, 
        (unsigned long long int *)bitset_trim.get_vec().data());
  #endif
    return bitset_trim;
  }

  // inclusive range
  static void reset_range(size_t begin, size_t end) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      bitset_trim_reset_cuda(cuda_ctx, begin, end);
    } else
  #endif
      { assert (personality == CPU);
      bitset_trim.reset(begin, end); }
  }
};
