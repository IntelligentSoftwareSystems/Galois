/**sync_structures helpers -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * sync_structures helper functions (see .h for macros)
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#include <galois/runtime/sync_structures.h>

// TODO put in a sensible namespace
bool src_invalid(BITVECTOR_STATUS bv_flag) {
  return (bv_flag == BITVECTOR_STATUS::SRC_INVALID || 
          bv_flag == BITVECTOR_STATUS::BOTH_INVALID);
}

bool dst_invalid(BITVECTOR_STATUS bv_flag) {
  return (bv_flag == BITVECTOR_STATUS::DST_INVALID || 
          bv_flag == BITVECTOR_STATUS::BOTH_INVALID);
}

void make_src_invalid(BITVECTOR_STATUS* bv_flag) {
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

void make_dst_invalid(BITVECTOR_STATUS* bv_flag) {
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
