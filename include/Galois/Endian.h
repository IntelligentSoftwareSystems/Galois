/** Endian utility functions -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_ENDIAN_H
#define GALOIS_ENDIAN_H

#include "Galois/config.h"
#include <stdint.h>

#define USE_NAIVE_BYTE_SWAP

#ifdef HAVE_ENDIAN_H
# include <endian.h>
#endif

namespace Galois {

// NB: Wrap these standard functions with different names because
// sometimes le64toh and such are implemented as macros and we don't
// want any nasty surprises.
static inline uint64_t convert_le64(uint64_t x) {
#if !defined(HAVE_BIG_ENDIAN)
  return x;
#elif defined(USE_NAIVE_BYTE_SWAP) || !defined(HAVE_LE64TOH)
  return ((x<<56) & 0xFF00000000000000) | 
         ((x<<40) & 0x00FF000000000000) |
         ((x<<24) & 0x0000FF0000000000) |
         ((x<<8 ) & 0x000000FF00000000) |
         ((x>>8 ) & 0x00000000FF000000) |
         ((x>>24) & 0x0000000000FF0000) |
         ((x>>40) & 0x000000000000FF00) |
         ((x>>56) & 0x00000000000000FF);
#else
  return le64toh(x);
#endif
}

static inline uint32_t convert_le32(uint32_t x) {
#if !defined(HAVE_BIG_ENDIAN)
  return x;
#elif defined(USE_NAIVE_BYTE_SWAP) || !defined(HAVE_LE64TOH)
  return ((x<<24) & 0xFF000000) |
         ((x<<8 ) & 0x00FF0000) |
         ((x>>8 ) & 0x0000FF00) |
         ((x>>24) & 0x000000FF);
#else
  return le32toh(x);
#endif
}

}
#endif
