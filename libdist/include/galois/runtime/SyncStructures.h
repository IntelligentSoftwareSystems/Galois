/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#ifndef _SYNC_STRUCT_MACROS_
#define _SYNC_STRUCT_MACROS_

#include <cstdint> // for uint types used below
#include <galois/AtomicHelpers.h> // for galois::max, min
#include <galois/gIO.h> // for GALOIS DIE

////////////////////////////////////////////////////////////////////////////////
// Field flag class
////////////////////////////////////////////////////////////////////////////////

namespace galois {
namespace runtime {

enum BITVECTOR_STATUS {
  NONE_INVALID,
  SRC_INVALID,
  DST_INVALID,
  BOTH_INVALID
};

bool src_invalid(BITVECTOR_STATUS bv_flag);
bool dst_invalid(BITVECTOR_STATUS bv_flag);
void make_src_invalid(BITVECTOR_STATUS* bv_flag);
void make_dst_invalid(BITVECTOR_STATUS* bv_flag);

/**
 * Each field has a FieldFlags object that indicates synchronization status
 * of that field.
 */
class FieldFlags {
 private:
  uint8_t _s2s;
  uint8_t _s2d;
  uint8_t _d2s;
  uint8_t _d2d;
 public:
  BITVECTOR_STATUS bitvectorStatus;
  /**
   * Field Flags constructor. Sets all flags to false
   */
  FieldFlags() {
    _s2s = false;
    _s2d = false;
    _d2s = false;
    _d2d = false;
    bitvectorStatus = BITVECTOR_STATUS::NONE_INVALID;
  }

  bool src_to_src() const {
    return _s2s;
  }

  bool src_to_dst() const {
    return _s2d;
  }

  bool dst_to_src() const {
    return _d2s;
  }

  bool dst_to_dst() const {
    return _d2d;
  }

  void set_write_src() {
    _s2s = true;
    _s2d = true;
  }

  void set_write_dst() {
    _d2s = true;
    _d2d = true;
  }

  void set_write_any() {
    _s2s = true;
    _s2d = true;
    _d2s = true;
    _d2d = true;
  }

  void clear_read_src() {
    _s2s = false;
    _d2s = false;
  }

  void clear_read_dst() {
    _s2d = false;
    _d2d = false;
  }

  void clear_read_any() {
    _s2d = false;
    _d2d = false;
    _s2s = false;
    _d2s = false;
  }
  
  void clear_all() {
    _s2s = false;
    _s2d = false;
    _d2s = false;
    _d2d = false;
    bitvectorStatus = BITVECTOR_STATUS::NONE_INVALID;
  }
};

} // end namespace runtime
} // end namespace galois

////////////////////////////////////////////////////////////////////////////////
// Reduce Add
////////////////////////////////////////////////////////////////////////////////

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_REDUCE_ADD(fieldname, fieldtype) \
struct Reduce_add_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  uint64_t *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_reset_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                              data_mode, (ValTy)0);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_reset_node_##fieldname##_cuda(cuda_ctx, from_id, y, (ValTy)0);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    if (personality == GPU_CUDA) {\
      batch_reset_node_##fieldname##_cuda(cuda_ctx, begin, end, (ValTy)0);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    if (personality == GPU_CUDA) {\
      add_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
      return true;\
    }\
    assert(personality == CPU);\
    { galois::add(node.fieldname, y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_add_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                         data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
    if (personality == GPU_CUDA) {\
      set_node_##fieldname##_cuda(cuda_ctx, node_id, (ValTy)0);\
    }\
    else if (personality == CPU)\
    galois::set(node.fieldname, (ValTy)0);\
  }\
}
#else
// Non-GPU code
#define GALOIS_SYNC_STRUCTURE_REDUCE_ADD(fieldname, fieldtype) \
struct Reduce_add_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  uint64_t *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    { galois::add(node.fieldname, y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
    galois::set(node.fieldname, (ValTy)0);\
  }\
}
#endif

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_REDUCE_ADD_ARRAY(fieldname, fieldtype) \
struct Reduce_add_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return fieldname[node_id];\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  uint64_t *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_reset_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                              data_mode, (ValTy)0);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_reset_node_##fieldname##_cuda(cuda_ctx, from_id, y, (ValTy)0);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    if (personality == GPU_CUDA) {\
      batch_reset_node_##fieldname##_cuda(cuda_ctx, begin, end, (ValTy)0);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    if (personality == GPU_CUDA) {\
      add_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
      return true;\
    }\
    assert(personality == CPU);\
    { galois::add(fieldname[node_id], y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_add_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                         data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
    if (personality == GPU_CUDA) {\
      set_node_##fieldname##_cuda(cuda_ctx, node_id, (ValTy)0);\
    }\
    else if (personality == CPU)\
    galois::set(fieldname[node_id], (ValTy)0);\
  }\
}
#else
// Non-GPU code
#define GALOIS_SYNC_STRUCTURE_REDUCE_ADD_ARRAY(fieldname, fieldtype) \
struct Reduce_add_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    return fieldname[node_id];\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  uint64_t *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    { galois::add(fieldname[node_id], y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
    galois::set(fieldname[node_id], (ValTy)0);\
  }\
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Reduce Set
////////////////////////////////////////////////////////////////////////////////

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_REDUCE_SET(fieldname, fieldtype) \
struct Reduce_set_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  uint64_t *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                              data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, y);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    return true;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    if (personality == GPU_CUDA) {\
      set_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
      return true;\
    }\
    assert(personality == CPU);\
    { galois::set(node.fieldname, y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_set_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                         data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#else
// Non-GPU code
#define GALOIS_SYNC_STRUCTURE_REDUCE_SET(fieldname, fieldtype) \
struct Reduce_set_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  uint64_t *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    return true;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    { galois::set(node.fieldname, y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#endif

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_REDUCE_SET_ARRAY(fieldname, fieldtype) \
struct Reduce_set_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return fieldname[node_id];\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  uint64_t *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                              data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, y);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    return true;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    if (personality == GPU_CUDA) {\
      set_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
      return true;\
    }\
    assert(personality == CPU);\
    { galois::set(fieldname[node_id], y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_set_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                         data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#else
// Non-GPU code
#define GALOIS_SYNC_STRUCTURE_REDUCE_SET_ARRAY(fieldname, fieldtype) \
struct Reduce_set_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    return fieldname[node_id];\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  uint64_t *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    return true;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    { galois::set(fieldname[node_id], y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Reduce Min
////////////////////////////////////////////////////////////////////////////////

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_REDUCE_MIN(fieldname, fieldtype) \
struct Reduce_min_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  uint64_t *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                              data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, y);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    return true;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    if (personality == GPU_CUDA) {\
      return y < min_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
    }\
    assert(personality == CPU);\
    { return y < galois::min(node.fieldname, y); }\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_min_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                        data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#else
// Non-GPU code
#define GALOIS_SYNC_STRUCTURE_REDUCE_MIN(fieldname, fieldtype) \
struct Reduce_min_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  uint64_t *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    return true;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    { return y < galois::min(node.fieldname, y); }\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#endif


////////////////////////////////////////////////////////////////////////////////
// Reduce Max
////////////////////////////////////////////////////////////////////////////////

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_REDUCE_MAX(fieldname, fieldtype) \
struct Reduce_max_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  uint64_t *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                              data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, y);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    return true;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    if (personality == GPU_CUDA) {\
      return y > max_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
    }\
    assert(personality == CPU);\
    { return y > galois::max(node.fieldname, y); }\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_max_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                        data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#else
// Non-GPU code
#define GALOIS_SYNC_STRUCTURE_REDUCE_MAX(fieldname, fieldtype) \
struct Reduce_max_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  uint64_t *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    return true;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    { return y > galois::max(node.fieldname, y); }\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#endif


// Sync structure used for reduce via min of all pairwise elements in an 
// array/vector type.
#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_REDUCE_MIN_ARRAY(fieldname, fieldtype) \
struct Reduce_min_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return fieldname[node_id];\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  uint64_t *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                              data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, y);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    return true;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    if (personality == GPU_CUDA) {\
      return y < min_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
    }\
    assert(personality == CPU);\
    { return y < galois::min(fieldname[node_id], y); }\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_min_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                        data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#else
// Non-GPU code
#define GALOIS_SYNC_STRUCTURE_REDUCE_MIN_ARRAY(fieldname, fieldtype) \
struct Reduce_min_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    return fieldname[node_id];\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  uint64_t *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    return true;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    { return y < galois::min(fieldname[node_id], y); }\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
  }\
}
#endif

// Sync structure used for reduce via average of all pairwise elements in an 
// array/vector type.
#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_AVG_ARRAY(fieldname, fieldtype) \
struct Reduce_pair_wise_avg_array_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  uint64_t *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                              data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, y);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    if (personality == GPU_CUDA) {\
      set_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
      return true;\
    }\
    assert(personality == CPU);\
    { galois::pairWiseAvg_vec(node.fieldname, y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_set_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                         data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
    { galois::resetVec(node.fieldname); }\
  }\
}
#else
// Non-GPU code
#define GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_AVG_ARRAY(fieldname, fieldtype) \
struct Reduce_pair_wise_avg_array_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  uint64_t *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    { galois::pairWiseAvg_vec(node.fieldname, y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
    { galois::resetVec(node.fieldname); }\
  }\
}
#endif

// Sync structure used for reduce via add of all pairwise elements in an 
// array/vector type.
#define GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_ADD_ARRAY(fieldname, fieldtype) \
struct Reduce_pair_wise_add_array_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node) {\
    return node.fieldname;\
  }\
\
  static bool extract_reset_batch(unsigned from_id,\
                                  uint64_t *b,\
                                  unsigned int *o,\
                                  ValTy *y,\
                                  size_t *s,\
                                  DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y) {\
    { galois::addArray(node.fieldname, y); return true;}\
  }\
\
  static bool reduce_batch(unsigned from_id,\
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
\
  static void reset (uint32_t node_id, struct NodeData &node) {\
    { galois::resetVec(node.fieldname); }\
  }\
}

// Sync structure used for reduce via average of a single element in an 
// array/vector type (requires an index).
#define GALOIS_SYNC_STRUCTURE_REDUCE_PAIR_WISE_ADD_ARRAY_SINGLE(fieldname, fieldtype) \
struct Reduce_pair_wise_add_array_single_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData &node,\
                       unsigned vecIndex) {\
    return node.fieldname[vecIndex];\
  }\
\
  static bool extract_reset_batch(unsigned, unsigned long long int*,\
                                  unsigned int*, ValTy*,\
                                  size_t*, DataCommMode*) {\
    return false;\
  }\
\
  static bool extract_reset_batch(unsigned, ValTy*) {\
    return false;\
  }\
\
  static bool reset_batch(size_t begin, size_t end) {\
    return false;\
  }\
\
  static bool reduce(uint32_t node_id, struct NodeData &node, ValTy y,\
                     unsigned vecIndex) {\
    node.fieldname[vecIndex] = node.fieldname[vecIndex] + y;\
    return true;\
  }\
\
  static bool reduce_batch(unsigned, unsigned long long int*,\
                           unsigned int*, ValTy *, size_t, DataCommMode) {\
    return false;\
  }\
\
  static void reset(uint32_t node_id, struct NodeData &node, unsigned vecIndex) {\
    node.fieldname[vecIndex] = 0;\
  }\
}



////////////////////////////////////////////////////////////////////////////////
// Broadcast
////////////////////////////////////////////////////////////////////////////////

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_BROADCAST(fieldname, fieldtype)\
struct Broadcast_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData & node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return node.fieldname;\
  }\
\
  static bool extract_batch(unsigned from_id,\
                            uint64_t *b,\
                            unsigned int *o,\
                            ValTy *y,\
                            size_t *s,\
                            DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                         data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_node_##fieldname##_cuda(cuda_ctx, from_id, y);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y) {\
    if (personality == GPU_CUDA)\
      set_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
    else if (personality == CPU)\
    node.fieldname = y;\
  }\
\
  static bool setVal_batch(unsigned from_id, \
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_set_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                               data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
}
#else
#define GALOIS_SYNC_STRUCTURE_BROADCAST(fieldname, fieldtype)\
struct Broadcast_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData & node) {\
    return node.fieldname;\
  }\
\
  static bool extract_batch(unsigned from_id,\
                            uint64_t *b,\
                            unsigned int *o,\
                            ValTy *y,\
                            size_t *s,\
                            DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y) {\
    node.fieldname = y;\
  }\
\
  static bool setVal_batch(unsigned from_id, \
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
}
#endif

// Sync structure used for broadcast synchronization of a single element in a 
// vector. The GALOIS_DIE functions are defined due a template path needing them
// defined (they shouldn't ever go down that path, however)
#define GALOIS_SYNC_STRUCTURE_BROADCAST_VECTOR_SINGLE(fieldname, fieldtype)\
struct Broadcast_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData & node,\
                       unsigned vecIndex) {\
    return node.fieldname[vecIndex];\
  }\
\
  static ValTy extract(uint32_t node_id, const struct NodeData & node) {\
    GALOIS_DIE("Execution shouldn't get here this function needs an index arg\n");\
    return node.fieldname[0];\
  }\
\
  static bool extract_batch(unsigned, unsigned long int *, unsigned int* ,\
                            ValTy *, size_t *, DataCommMode *) {\
    return false;\
  }\
\
  static bool extract_batch(unsigned, ValTy *) {\
    return false;\
  }\
\
  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y,\
                     unsigned vecIndex) {\
    node.fieldname[vecIndex] = y;\
  }\
\
  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y) {\
    GALOIS_DIE("Execution shouldn't get here; needs index arg\n");\
  }\
\
  static bool setVal_batch(unsigned, unsigned long int*, unsigned int*,\
                           ValTy*, size_t, DataCommMode) {\
    return false;\
  }\
}


// Used in a "struct of arrays" construction of data (i.e. for some field,
// index into it with the node id to get data)
#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_BROADCAST_ARRAY(fieldname, fieldtype)\
struct Broadcast_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData & node) {\
    if (personality == GPU_CUDA)\
      return get_node_##fieldname##_cuda(cuda_ctx, node_id);\
    assert (personality == CPU);\
    return fieldname[node_id];\
  }\
\
  static bool extract_batch(unsigned from_id,\
                            uint64_t *b,\
                            unsigned int *o,\
                            ValTy *y,\
                            size_t *s,\
                            DataCommMode *data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_get_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                         data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static bool extract_batch(unsigned from_id, ValTy *y) {\
    if (personality == GPU_CUDA) {\
      batch_get_node_##fieldname##_cuda(cuda_ctx, from_id, y);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
\
  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y) {\
    if (personality == GPU_CUDA)\
      set_node_##fieldname##_cuda(cuda_ctx, node_id, y);\
    else if (personality == CPU)\
    fieldname[node_id] = y;\
  }\
\
  static bool setVal_batch(unsigned from_id, \
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    if (personality == GPU_CUDA) {\
      batch_set_mirror_node_##fieldname##_cuda(cuda_ctx, from_id, b, o, y, s,\
                                               data_mode);\
      return true;\
    }\
    assert (personality == CPU);\
    return false;\
  }\
}
#else
#define GALOIS_SYNC_STRUCTURE_BROADCAST_ARRAY(fieldname, fieldtype)\
struct Broadcast_##fieldname {\
  typedef fieldtype ValTy;\
\
  static ValTy extract(uint32_t node_id, const struct NodeData & node) {\
    return fieldname[node_id];\
  }\
\
  static bool extract_batch(unsigned from_id,\
                            uint64_t *b,\
                            unsigned int *o,\
                            ValTy *y,\
                            size_t *s,\
                            DataCommMode *data_mode) {\
    return false;\
  }\
\
  static bool extract_batch(unsigned from_id, ValTy *y) {\
    return false;\
  }\
\
  static void setVal(uint32_t node_id, struct NodeData & node, ValTy y) {\
    fieldname[node_id] = y;\
  }\
\
  static bool setVal_batch(unsigned from_id, \
                           uint64_t *b,\
                           unsigned int *o,\
                           ValTy *y,\
                           size_t s,\
                           DataCommMode data_mode) {\
    return false;\
  }\
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Bitset struct
////////////////////////////////////////////////////////////////////////////////

// Important: Bitsets are expected to have the following naming scheme:
// bitset_<fieldname>
// In addition, you will have to declare and appropriately resize the bitset 
// in your main program as well as set the bitset appropriately (i.e. when you
// do a write to a particular node).

#ifdef __GALOIS_HET_CUDA__
// GPU code included
#define GALOIS_SYNC_STRUCTURE_BITSET(fieldname)\
struct Bitset_##fieldname {\
  static constexpr bool is_vector_bitset() {\
    return false;\
  }\
  static bool is_valid() {\
    return true;\
  }\
\
  static galois::DynamicBitSet& get() {\
    if (personality == GPU_CUDA) \
      get_bitset_##fieldname##_cuda(cuda_ctx,\
        (uint64_t *)bitset_##fieldname.get_vec().data());\
    return bitset_##fieldname;\
  }\
\
  static void reset_range(size_t begin, size_t end) {\
    if (personality == GPU_CUDA) {\
      bitset_##fieldname##_reset_cuda(cuda_ctx, begin, end);\
    } else\
      { assert (personality == CPU);\
      bitset_##fieldname.reset(begin, end); }\
  }\
}
#else
// no GPU code
#define GALOIS_SYNC_STRUCTURE_BITSET(fieldname)\
struct Bitset_##fieldname {\
  static constexpr bool is_vector_bitset() {\
    return false;\
  }\
\
  static constexpr bool is_valid() {\
    return true;\
  }\
\
  static galois::DynamicBitSet& get() {\
    return bitset_##fieldname;\
  }\
\
  static void reset_range(size_t begin, size_t end) {\
    bitset_##fieldname.reset(begin, end);\
  }\
}
#endif

// Used for a vector of bitsets: function signatures now allow indexing into
// this vector to get the correct bitset
#define GALOIS_SYNC_STRUCTURE_VECTOR_BITSET(fieldname)\
struct Bitset_##fieldname {\
  static unsigned numBitsets() {\
    return vbitset_##fieldname.size();\
  }\
\
  static constexpr bool is_vector_bitset() {\
    return true;\
  }\
\
  static constexpr bool is_valid() {\
    return true;\
  }\
\
  static galois::DynamicBitSet& get(unsigned i) {\
    return vbitset_##fieldname[i];\
  }\
\
  static void reset_range(size_t begin, size_t end) {\
    for (unsigned i = 0; i < vbitset_##fieldname.size(); i++) {\
      vbitset_##fieldname[i].reset(begin, end);\
    }\
  }\
}

#endif // header guard
