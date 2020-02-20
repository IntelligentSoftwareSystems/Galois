/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_ENDIAN_H
#define GALOIS_ENDIAN_H

#include <cstdint>

namespace galois {

static inline uint32_t bswap32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_bswap32(x);
#else
  return ((x << 24) & 0xff000000) | ((x << 8) & 0x00ff0000) |
         ((x >> 8) & 0x0000ff00) | ((x >> 24) & 0x000000ff);
#endif
}

static inline uint64_t bswap64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_bswap64(x);
#else
  return ((x << 56) & 0xff00000000000000UL) |
         ((x << 40) & 0x00ff000000000000UL) |
         ((x << 24) & 0x0000ff0000000000UL) |
         ((x <<  8) & 0x000000ff00000000UL) |
         ((x >>  8) & 0x00000000ff000000UL) |
         ((x >> 24) & 0x0000000000ff0000UL) |
         ((x >> 40) & 0x000000000000ff00UL) |
         ((x >> 56) & 0x00000000000000ffUL);
#endif
}

// NB: Wrap these standard functions with different names because
// sometimes le64toh and such are implemented as macros and we don't
// want any nasty surprises.
static inline uint64_t convert_le64toh(uint64_t x) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return x;
#else
  return bswap64(x);
#endif
}

static inline uint32_t convert_le32toh(uint32_t x) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return x;
#else
  return bswap32(x);
#endif
}

static inline uint64_t convert_htobe64(uint64_t x) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return x;
#else
  return bswap64(x);
#endif
}

static inline uint32_t convert_htobe32(uint32_t x) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return x;
#else
  return bswap32(x);
#endif
}

static inline uint64_t convert_htole64(uint64_t x) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return x;
#else
  return bswap64(x);
#endif
}

static inline uint32_t convert_htole32(uint32_t x) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return x;
#else
  return bswap32(x);
#endif
}

} // namespace galois
#endif
