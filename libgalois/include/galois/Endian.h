#ifndef GALOIS_ENDIAN_H
#define GALOIS_ENDIAN_H

#include <cstdint>
#ifndef _BSD_SOURCE
#define _BSD_SOURCE 1
#endif
#include <endian.h>

namespace galois {

// NB: Wrap these standard functions with different names because
// sometimes le64toh and such are implemented as macros and we don't
// want any nasty surprises.
static inline uint64_t convert_le64toh(uint64_t x) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return x;
#else
  return le64toh(x);
#endif
}

static inline uint32_t convert_le32toh(uint32_t x) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return x;
#else
  return le32toh(x);
#endif
}

static inline uint64_t convert_htobe64(uint64_t x) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return x;
#else
  return htobe64(x);
#endif
}

static inline uint32_t convert_htobe32(uint32_t x) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return x;
#else
  return htobe32(x);
#endif
}

static inline uint64_t convert_htole64(uint64_t x) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return x;
#else
  return htole64(x);
#endif
}

static inline uint32_t convert_htole32(uint32_t x) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return x;
#else
  return htole32(x);
#endif
}


static inline uint32_t bswap32(uint32_t x) {
  return
    ((x << 24) & 0xff000000 ) |
    ((x <<  8) & 0x00ff0000 ) |
    ((x >>  8) & 0x0000ff00 ) |
    ((x >> 24) & 0x000000ff );
}

static inline uint64_t bswap64(uint64_t x) {
  return
    ( (x << 56) & 0xff00000000000000UL ) |
    ( (x << 40) & 0x00ff000000000000UL ) |
    ( (x << 24) & 0x0000ff0000000000UL ) |
    ( (x <<  8) & 0x000000ff00000000UL ) |
    ( (x >>  8) & 0x00000000ff000000UL ) |
    ( (x >> 24) & 0x0000000000ff0000UL ) |
    ( (x >> 40) & 0x000000000000ff00UL ) |
    ( (x >> 56) & 0x00000000000000ffUL );
}

}
#endif
