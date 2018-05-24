#include <galois/runtime/SyncStructures.h>

using namespace galois::runtime; // for easy access to BITVECTOR_STATUS

bool galois::runtime::src_invalid(BITVECTOR_STATUS bv_flag) {
  return (bv_flag == BITVECTOR_STATUS::SRC_INVALID || 
          bv_flag == BITVECTOR_STATUS::BOTH_INVALID);
}

bool galois::runtime::dst_invalid(BITVECTOR_STATUS bv_flag) {
  return (bv_flag == BITVECTOR_STATUS::DST_INVALID || 
          bv_flag == BITVECTOR_STATUS::BOTH_INVALID);
}

void galois::runtime::make_src_invalid(BITVECTOR_STATUS* bv_flag) {
  switch(*bv_flag) {
    case NONE_INVALID:
      *bv_flag = BITVECTOR_STATUS::SRC_INVALID;
      break;
    case DST_INVALID:
      *bv_flag = BITVECTOR_STATUS::BOTH_INVALID;
      break;
    case SRC_INVALID:
    case BOTH_INVALID:
      break;
  }
}

void galois::runtime::make_dst_invalid(BITVECTOR_STATUS* bv_flag) {
  switch(*bv_flag) {
    case NONE_INVALID:
      *bv_flag = BITVECTOR_STATUS::DST_INVALID;
      break;
    case SRC_INVALID:
      *bv_flag = BITVECTOR_STATUS::BOTH_INVALID;
      break;
    case DST_INVALID:
    case BOTH_INVALID:
      break;
  }
}
