////////////////////////////////////////////////////////////////////////////
// # short paths
////////////////////////////////////////////////////////////////////////////
struct Reduce_num_shortest_paths {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_num_shortest_paths_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.num_shortest_paths;
  }

  static bool extract_reset_batch(unsigned from_id, 
                                  unsigned long long int *b, 
                                  unsigned int *o, unsigned int *y, 
                                  size_t *s, DataCommMode *data_mode) {

  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { 
      batch_get_reset_node_num_shortest_paths_cuda(cuda_ctx, from_id, b, o,
                                                   y, s, data_mode, 0);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { 
      batch_get_reset_node_num_shortest_paths_cuda(cuda_ctx, from_id, y, 0);
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
      add_node_num_shortest_paths_cuda(cuda_ctx, node_id, y);
      return true;
    } 
    //else if (personality == CPU)
    assert(personality == CPU);
  #endif
    Galois::add(node.num_shortest_paths, y); return true;
  }

  static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                           unsigned int *o, unsigned int *y, size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_add_node_num_shortest_paths_cuda(cuda_ctx, from_id, b, o, y, s, 
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
      set_node_num_shortest_paths_cuda(cuda_ctx, node_id, 0);
    }
    else if (personality == CPU)
  #endif
    Galois::set(node.num_shortest_paths, (unsigned int)0);
  }
};

struct ReduceSet_num_shortest_paths {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_num_shortest_paths_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.num_shortest_paths;
  }

  static bool extract_reset_batch(unsigned from_id, 
                                  unsigned long long int *b, 
                                  unsigned int *o, 
                                  unsigned int *y, 
                                  size_t *s, 
                                  DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { 
      batch_get_mirror_node_num_shortest_paths_cuda(cuda_ctx, from_id, b, o, y, s, 
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
      batch_get_mirror_node_num_shortest_paths_cuda(cuda_ctx, from_id, y);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool reduce(uint32_t node_id, struct NodeData & node, 
                     unsigned int y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      set_node_num_shortest_paths_cuda(cuda_ctx, node_id, y);
    }
    else if (personality == CPU)
  #endif
    Galois::set(node.num_shortest_paths, y);

    return true;
  }

  static bool reduce_batch(unsigned from_id, 
                           unsigned long long int *b, 
                           unsigned int *o, 
                           unsigned int *y, 
                           size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_node_num_shortest_paths_cuda(cuda_ctx, from_id, b, o, y, s, 
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


struct Broadcast_num_shortest_paths {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_num_shortest_paths_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif

    return node.num_shortest_paths;
  }

  static bool extract_batch(unsigned from_id, unsigned long long int *b,
                            unsigned int *o, unsigned int *y, size_t *s, 
                            DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_node_num_shortest_paths_cuda(cuda_ctx, from_id, b, o, y, s,
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
      batch_get_node_num_shortest_paths_cuda(cuda_ctx, from_id, y);
      return true;
    }
    assert (personality == CPU);
  #endif

    return false;
  }

  static void setVal(uint32_t node_id, struct NodeData & node, 
                     ValTy y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA)
      set_node_num_shortest_paths_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
    Galois::set(node.num_shortest_paths, y);
  }

  static bool setVal_batch(unsigned from_id, unsigned long long int *b, 
                           unsigned int *o, unsigned int *y, size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_node_num_shortest_paths_cuda(cuda_ctx, from_id, b, o, y, s,
                               data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }
};

////////////////////////////////////////////////////////////////////////////
// Succ
////////////////////////////////////////////////////////////////////////////
struct Reduce_num_successors {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_num_successors_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.num_successors;
  }

  static bool extract_reset_batch(unsigned from_id, 
                                  unsigned long long int *b, 
                                  unsigned int *o, 
                                  unsigned int *y, 
                                  size_t *s, 
                                  DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { 
      batch_get_reset_node_num_successors_cuda(cuda_ctx, from_id, b, o, y, s, 
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
      batch_get_reset_node_num_successors_cuda(cuda_ctx, from_id, y, 0);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool reduce(uint32_t node_id, struct NodeData & node, 
                     unsigned int y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      add_node_num_successors_cuda(cuda_ctx, node_id, y);
      return true;
    } 
    //else if (personality == CPU)
    assert(personality == CPU);
  #endif
    { return Galois::add(node.num_successors, y); return true; }
  }

  static bool reduce_batch(unsigned from_id, 
                           unsigned long long int *b, 
                           unsigned int *o, 
                           unsigned int *y, 
                           size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_add_node_num_successors_cuda(cuda_ctx, from_id, b, o, y, s, 
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
      set_node_num_successors_cuda(cuda_ctx, node_id, 0);
    }
    else if (personality == CPU)
  #endif
    Galois::set(node.num_successors, (unsigned int)0);
  }
};

struct ReduceSet_num_successors {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_num_successors_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.num_successors;
  }

  static bool extract_reset_batch(unsigned from_id, 
                                  unsigned long long int *b, 
                                  unsigned int *o, 
                                  unsigned int *y, 
                                  size_t *s, 
                                  DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { 
      batch_get_mirror_node_num_successors_cuda(cuda_ctx, from_id, b, o, y, s, 
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
      batch_get_mirror_node_num_successors_cuda(cuda_ctx, from_id, y);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool reduce(uint32_t node_id, struct NodeData & node, 
                     unsigned int y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      set_node_num_successors_cuda(cuda_ctx, node_id, y);
    }
    else if (personality == CPU)
  #endif
    Galois::set(node.num_successors, y);

    return true;
  }

  static bool reduce_batch(unsigned from_id, 
                           unsigned long long int *b, 
                           unsigned int *o, 
                           unsigned int *y, 
                           size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_node_num_successors_cuda(cuda_ctx, from_id, b, o, y, s, 
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


struct Broadcast_num_successors {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_num_successors_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.num_successors;
  }

  static bool extract_batch(unsigned from_id,
                            unsigned long long int *b,
                            unsigned int *o,
                            unsigned int *y,
                            size_t *s, 
                            DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_node_num_successors_cuda(cuda_ctx, from_id, b, o, y, s,
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
      batch_get_node_num_successors_cuda(cuda_ctx, from_id, y);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static void setVal(uint32_t node_id, struct NodeData & node, 
                     unsigned int y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA)
      set_node_num_successors_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
    Galois::set(node.num_successors, y);
  }

  static bool setVal_batch(unsigned from_id, 
                           unsigned long long int *b, 
                           unsigned int *o, 
                           unsigned int *y, 
                           size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_node_num_successors_cuda(cuda_ctx, from_id, b, o, y, s, 
                               data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }
};

////////////////////////////////////////////////////////////////////////////
// Pred
////////////////////////////////////////////////////////////////////////////

struct Reduce_num_predecessors {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_num_predecessors_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.num_predecessors;
  }

  static bool extract_reset_batch(unsigned from_id, 
                                  unsigned long long int *b, 
                                  unsigned int *o, 
                                  unsigned int *y, 
                                  size_t *s, 
                                  DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { 
      batch_get_reset_node_num_predecessors_cuda(cuda_ctx, from_id, b, o, y, s, 
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
      batch_get_reset_node_num_predecessors_cuda(cuda_ctx, from_id, y, 0);
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
      add_node_num_predecessors_cuda(cuda_ctx, node_id, y);
      return true;
    } 
    //else if (personality == CPU)
    assert(personality == CPU);
  #endif
    { return Galois::add(node.num_predecessors, y); return true; }
  }

  static bool reduce_batch(unsigned from_id, 
                           unsigned long long int *b, 
                           unsigned int *o, 
                           unsigned int *y, 
                           size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_add_node_num_predecessors_cuda(cuda_ctx, from_id, b, o, y, s, 
                               data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static void reset(uint32_t node_id, struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      set_node_num_predecessors_cuda(cuda_ctx, node_id, 0);
    }
    else if (personality == CPU)
  #endif
    Galois::set(node.num_predecessors, (unsigned int)0);
  }
};

struct ReduceSet_num_predecessors {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_num_predecessors_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.num_predecessors;
  }

  static bool extract_reset_batch(unsigned from_id, 
                                  unsigned long long int *b, 
                                  unsigned int *o, 
                                  unsigned int *y, 
                                  size_t *s, 
                                  DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { 
      batch_get_mirror_node_num_predecessors_cuda(cuda_ctx, from_id, b, o, y, s, 
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
      batch_get_mirror_node_num_predecessors_cuda(cuda_ctx, from_id, y);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool reduce(uint32_t node_id, struct NodeData & node, 
                     unsigned int y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      set_node_num_predecessors_cuda(cuda_ctx, node_id, y);
    }
    else if (personality == CPU)
  #endif
    Galois::set(node.num_predecessors, y);

    return true;
  }

  static bool reduce_batch(unsigned from_id, 
                           unsigned long long int *b, 
                           unsigned int *o, 
                           unsigned int *y, 
                           size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_node_num_predecessors_cuda(cuda_ctx, from_id, b, o, y, s, 
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


struct Broadcast_num_predecessors {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_num_predecessors_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.num_predecessors;
  }

  static bool extract_batch(unsigned from_id,
                            unsigned long long int *b,
                            unsigned int *o,
                            unsigned int *y,
                            size_t *s, 
                            DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_node_num_predecessors_cuda(cuda_ctx, from_id, b, o, y, s,
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
      batch_get_node_num_predecessors_cuda(cuda_ctx, from_id, y);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static void setVal(uint32_t node_id, struct NodeData & node, 
                     ValTy y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA)
      set_node_num_predecessors_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
    Galois::set(node.num_predecessors, y);
  }

  static bool setVal_batch(unsigned from_id, 
                           unsigned long long int *b, 
                           unsigned int *o, 
                           unsigned int *y, 
                           size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_node_num_predecessors_cuda(cuda_ctx, from_id, b, o, y, s, 
                                           data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }
};

////////////////////////////////////////////////////////////////////////////
// Trim
////////////////////////////////////////////////////////////////////////////
struct Reduce_trim {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_trim_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.trim;
  }

  static bool extract_reset_batch(unsigned from_id, 
                                  unsigned long long int *b, 
                                  unsigned int *o, unsigned int *y, 
                                  size_t *s, DataCommMode *data_mode) {
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

  static bool reduce(uint32_t node_id, struct NodeData & node, 
                     ValTy y) {
  #ifdef __GALOIS_HET_CUDA__

    if (personality == GPU_CUDA) {
      add_node_trim_cuda(cuda_ctx, node_id, y);
      return true;
    } 
    //else if (personality == CPU)
    assert(personality == CPU);
  #endif
    { Galois::add(node.trim, y); return true; }
  }

  static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                           unsigned int *o, unsigned int *y, size_t s, 
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

struct ReduceSet_trim {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, 
                              const struct NodeData & node) {
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
      batch_get_mirror_node_trim_cuda(cuda_ctx, from_id, b, o, y, s, data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_mirror_node_trim_cuda(cuda_ctx, from_id, y);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool reduce(uint32_t node_id, struct NodeData & node, 
                     unsigned int y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      set_node_trim_cuda(cuda_ctx, node_id, y);
    }
    else if (personality == CPU)
  #endif
    Galois::set(node.trim, y);

    return true;
  }

  static bool reduce_batch(unsigned from_id, 
                           unsigned long long int *b, 
                           unsigned int *o, 
                           unsigned int *y, 
                           size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_node_trim_cuda(cuda_ctx, from_id, b, o, y, s, 
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


struct Broadcast_trim {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_trim_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.trim;
  }

  static bool extract_batch(unsigned from_id, unsigned long long int *b,
                            unsigned int *o, unsigned int *y, size_t *s, 
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

  static void setVal(uint32_t node_id, struct NodeData & node, 
                     ValTy y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA)
      set_node_trim_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
    Galois::set(node.trim, y);
  }

  static bool setVal_batch(unsigned from_id, unsigned long long int *b, 
                           unsigned int *o, unsigned int *y, size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_node_trim_cuda(cuda_ctx, from_id, b, o, y, s, 
                               data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }
};
////////////////////////////////////////////////////////////////////////////
// Current Lengths
////////////////////////////////////////////////////////////////////////////
struct Reduce_current_length {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_current_length_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.current_length;
  }

  static bool extract_reset_batch(unsigned from_id, 
                                  unsigned long long int *b, 
                                  unsigned int *o, unsigned int *y, 
                                  size_t *s, DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { 
      batch_get_mirror_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s, 
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
      batch_get_mirror_node_current_length_cuda(cuda_ctx, from_id, y);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool reduce(uint32_t node_id, struct NodeData & node, ValTy y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      return min_node_current_length_cuda(cuda_ctx, node_id, y);
      //return true;
    } 
    //else if (personality == CPU)
    assert(personality == CPU);
  #endif
    return y < Galois::min(node.current_length, y);
  }

  static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                           unsigned int *o, unsigned int *y, size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_min_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s, 
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

struct ReduceSet_current_length {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_current_length_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.current_length;
  }

  static bool extract_reset_batch(unsigned from_id, 
                                  unsigned long long int *b, 
                                  unsigned int *o, unsigned int *y, 
                                  size_t *s, DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { 
      batch_get_mirror_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s, 
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
      batch_get_mirror_node_current_length_cuda(cuda_ctx, from_id, y);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool reduce(uint32_t node_id, struct NodeData & node, 
                     ValTy y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA)
      set_node_current_length_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
    Galois::set(node.current_length, y);
    return true;
  }

  static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                           unsigned int *o, unsigned int *y, size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_node_current_length_cuda(cuda_ctx, from_id, b, o, 
                                     y, s, data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static void reset (uint32_t node_id, struct NodeData & node) {
    // NO RESET
  }
};


struct Broadcast_current_length {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_current_length_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.current_length;
  }

  static bool extract_batch(unsigned from_id,
                            unsigned long long int *b,
                            unsigned int *o,
                            unsigned int *y,
                            size_t *s, 
                            DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s,
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
      batch_get_node_current_length_cuda(cuda_ctx, from_id, y);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static void setVal(uint32_t node_id, struct NodeData & node, 
                     ValTy y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA)
      set_node_current_length_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
    Galois::set(node.current_length, y);
  }

  static bool setVal_batch(unsigned from_id, 
                           unsigned long long int *b, 
                           unsigned int *o, 
                           unsigned int *y, 
                           size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_node_current_length_cuda(cuda_ctx, from_id, b, o, y, s, 
                               data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Old length
////////////////////////////////////////////////////////////////////////////////

struct ReduceSet_old_length {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_old_length_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.old_length;
  }

  static bool extract_reset_batch(unsigned from_id, 
                                  unsigned long long int *b, 
                                  unsigned int *o, unsigned int *y, 
                                  size_t *s, DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { 
      batch_get_mirror_node_old_length_cuda(cuda_ctx, from_id, b, o, y, s, 
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
      batch_get_mirror_node_old_length_cuda(cuda_ctx, from_id, y);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool reduce(uint32_t node_id, struct NodeData & node, 
                     ValTy y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA)
      set_node_old_length_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
    Galois::set(node.old_length, y);
    return true;
  }

  static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                           unsigned int *o, unsigned int *y, size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_node_old_length_cuda(cuda_ctx, from_id, b, o, 
                                     y, s, data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static void reset (uint32_t node_id, struct NodeData & node) {
    // NO RESET
  }
};

struct Broadcast_old_length {
  typedef unsigned int ValTy;

  static unsigned int extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_old_length_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.old_length;
  }

  static bool extract_batch(unsigned from_id,
                            unsigned long long int *b,
                            unsigned int *o,
                            unsigned int *y,
                            size_t *s, 
                            DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_node_old_length_cuda(cuda_ctx, from_id, b, o, y, s,
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
      batch_get_node_old_length_cuda(cuda_ctx, from_id, y);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static void setVal(uint32_t node_id, struct NodeData & node, 
                     ValTy y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA)
      set_node_old_length_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
    Galois::set(node.old_length, y);
  }

  static bool setVal_batch(unsigned from_id, 
                           unsigned long long int *b, 
                           unsigned int *o, 
                           unsigned int *y, 
                           size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_node_old_length_cuda(cuda_ctx, from_id, b, o, y, s, 
                               data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }
};


////////////////////////////////////////////////////////////////////////////
// Flag
////////////////////////////////////////////////////////////////////////////
struct ReduceSet_propogation_flag {
  typedef unsigned int ValTy;

  static bool extract(uint32_t node_id, const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_propogation_flag_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.propogation_flag;
  }

  static bool extract_reset_batch(unsigned from_id, 
                                  unsigned long long int *b, 
                                  unsigned int *o, unsigned int *y, 
                                  size_t *s, DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { 
      batch_get_mirror_node_propogation_flag_cuda(cuda_ctx, from_id, b, o, 
                                                  (bool*)y, s, data_mode);
                                                  
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool extract_reset_batch(unsigned from_id, unsigned int *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_mirror_node_propogation_flag_cuda(cuda_ctx, from_id, (bool*)y);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool reduce(uint32_t node_id, struct NodeData & node, 
                     ValTy y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA)
      set_node_propogation_flag_cuda(cuda_ctx, node_id, (bool)y);
    else if (personality == CPU)
  #endif
    Galois::set(node.propogation_flag, (bool)y);
    return true;
  }

  static bool reduce_batch(unsigned from_id, unsigned long long int *b, 
                           unsigned int *o, unsigned int *y, size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_node_propogation_flag_cuda(cuda_ctx, from_id, b, o, 
                                           (bool*)y, s, data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static void reset (uint32_t node_id, struct NodeData & node) {
    // NO RESET
  }
};

struct Broadcast_propogation_flag {
  typedef unsigned int ValTy;

  static bool extract(uint32_t node_id, const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_propogation_flag_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.propogation_flag;
  }

  static bool extract_batch(unsigned from_id, unsigned long long int *b,
                            unsigned int *o, unsigned int *y, size_t *s, 
                            DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_node_propogation_flag_cuda(cuda_ctx, from_id, b, o, 
                                          (bool*)y, s, data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool extract_batch(unsigned from_id, unsigned int *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_node_propogation_flag_cuda(cuda_ctx, from_id, (bool*)y);
      return true;
    }
    assert (personality == CPU);
  #endif

    return false;
  }

  static void setVal(uint32_t node_id, struct NodeData & node, bool y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA)
      set_node_propogation_flag_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
    Galois::set(node.propogation_flag, y);
  }

  static bool setVal_batch(unsigned from_id, unsigned long long int *b, 
                           unsigned int *o, ValTy *y, size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_node_propogation_flag_cuda(cuda_ctx, from_id, b, o, 
                                           (bool*)y, s, data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }
};

////////////////////////////////////////////////////////////////////////////
// Dependency
////////////////////////////////////////////////////////////////////////////

struct Reduce_dependency {
  typedef float ValTy;

  static float extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_dependency_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.dependency;
  }

  static bool extract_reset_batch(unsigned from_id, 
                                  unsigned long long int *b, 
                                  unsigned int *o, 
                                  float *y, 
                                  size_t *s, 
                                  DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { 
      batch_get_reset_node_dependency_cuda(cuda_ctx, from_id, b, o, y, s, 
                                     data_mode, 0);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool extract_reset_batch(unsigned from_id, float *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_reset_node_dependency_cuda(cuda_ctx, from_id, y, 0);
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
      add_node_dependency_cuda(cuda_ctx, node_id, y);
      return true;
    } 
    //else if (personality == CPU)
    assert(personality == CPU);
  #endif
    { return Galois::add(node.dependency, y); return true; }
  }

  static bool reduce_batch(unsigned from_id, 
                           unsigned long long int *b, 
                           unsigned int *o, 
                           float *y, 
                           size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {

      batch_add_node_dependency_cuda(cuda_ctx, from_id, b, o, y, s, 
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
      set_node_dependency_cuda(cuda_ctx, node_id, 0);
    }
    else if (personality == CPU)
  #endif
    Galois::set(node.dependency, (float)0);
  }
};

struct ReduceSet_dependency {
  typedef float ValTy;

  static ValTy extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_dependency_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.dependency;
  }

  static bool extract_reset_batch(unsigned from_id, 
                                  unsigned long long int *b, 
                                  unsigned int *o, 
                                  ValTy *y, 
                                  size_t *s, 
                                  DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) { 
      batch_get_mirror_node_dependency_cuda(cuda_ctx, from_id, b, o, y, s, 
                                     data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool extract_reset_batch(unsigned from_id, ValTy *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_mirror_node_dependency_cuda(cuda_ctx, from_id, y);
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
      set_node_dependency_cuda(cuda_ctx, node_id, y);
    }
    else if (personality == CPU)
  #endif
    Galois::set(node.dependency, y);

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
      batch_set_node_dependency_cuda(cuda_ctx, from_id, b, o, y, s, 
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



struct Broadcast_dependency {
  typedef float ValTy;

  static float extract(uint32_t node_id, 
                              const struct NodeData & node) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) 
      return get_node_dependency_cuda(cuda_ctx, node_id);
    assert (personality == CPU);
  #endif
    return node.dependency;
  }

  static bool extract_batch(unsigned from_id,
                            unsigned long long int *b,
                            unsigned int *o,
                            float *y,
                            size_t *s, 
                            DataCommMode *data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_node_dependency_cuda(cuda_ctx, from_id, b, o, y, s,
                               data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static bool extract_batch(unsigned from_id, float *y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_get_node_dependency_cuda(cuda_ctx, from_id, y);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }

  static void setVal(uint32_t node_id, struct NodeData & node, 
                     ValTy y) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA)
      set_node_dependency_cuda(cuda_ctx, node_id, y);
    else if (personality == CPU)
  #endif
    Galois::set(node.dependency, (float)y);
  }

  static bool setVal_batch(unsigned from_id, 
                           unsigned long long int *b, 
                           unsigned int *o, 
                           float *y, 
                           size_t s, 
                           DataCommMode data_mode) {
  #ifdef __GALOIS_HET_CUDA__
    if (personality == GPU_CUDA) {
      batch_set_node_dependency_cuda(cuda_ctx, from_id, b, o, y, s, 
                               data_mode);
      return true;
    }
    assert (personality == CPU);
  #endif
    return false;
  }
};

// TODO set reduce for all ops
