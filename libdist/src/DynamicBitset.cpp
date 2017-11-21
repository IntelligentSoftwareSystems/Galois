#include "galois/DynamicBitset.h"
#include <boost/iterator/counting_iterator.hpp>
#include "galois/GaloisForwardDecl.h"
#include "galois/Traits.h"
#include "galois/Galois.h"

namespace galois {

void DynamicBitSet::reset(size_t begin, size_t end) {
  assert(begin <= (num_bits - 1));
  assert(end <= (num_bits - 1));

  // 100% safe implementation, but slow
  //for (unsigned long i = begin; i <= end; i++) {
  //  size_t bit_index = i / bits_uint64;
  //  uint64_t bit_offset = 1;
  //  bit_offset <<= (i % bits_uint64);
  //  uint64_t mask = ~bit_offset;
  //  bitvec[bit_index] &= mask;
  //}

  // block which you are safe to clear
  size_t vec_begin = (begin + bits_uint64 - 1) / bits_uint64;
  size_t vec_end;

  if (end == (num_bits - 1)) vec_end = bitvec.size();
  else vec_end = (end+1)/bits_uint64; // floor

  if (vec_begin < vec_end) {
    std::fill(bitvec.begin() + vec_begin, bitvec.begin() + vec_end, 0);
  }

  vec_begin *= bits_uint64;
  vec_end *= bits_uint64;

  // at this point vec_begin -> vec_end-1 has been reset

  if (vec_begin > vec_end) {
    // no fill happened
    if (begin < vec_begin) {
      size_t diff = vec_begin - begin;
      assert(diff < 64);
      uint64_t mask = ((uint64_t)1 << (64 - diff)) - 1;

      size_t end_diff = end - vec_end + 1;
      uint64_t or_mask = ((uint64_t)1 << end_diff) - 1;
      mask |= ~or_mask;

      size_t bit_index = begin / bits_uint64;
      bitvec[bit_index] &= mask;
    }
  } else {
    if (begin < vec_begin) {
      size_t diff = vec_begin - begin;
      assert(diff < 64);
      uint64_t mask = ((uint64_t)1 << (64 - diff)) - 1;
      size_t bit_index = begin/bits_uint64;
      bitvec[bit_index] &= mask;
    }
    if (end >= vec_end) {
      size_t diff = end - vec_end + 1;
      assert(diff < 64);
      uint64_t mask = ((uint64_t)1 << diff) - 1;
      size_t bit_index = end/bits_uint64;
      bitvec[bit_index] &= ~mask;
    }
  }
}

// assumes bit_vector is not updated (set) in parallel
bool DynamicBitSet::test(size_t index) const {
  size_t bit_index = index/bits_uint64;
  uint64_t bit_offset = 1;
  bit_offset <<= (index%bits_uint64);
  return ((bitvec[bit_index] & bit_offset) != 0);
}

void DynamicBitSet::set(size_t index) {
  size_t bit_index = index/bits_uint64;
  uint64_t bit_offset = 1;
  bit_offset <<= (index%bits_uint64);
  if ((bitvec[bit_index] & bit_offset) == 0) { // test and set
    size_t old_val = bitvec[bit_index];
    while(!bitvec[bit_index].compare_exchange_weak(old_val, old_val | bit_offset, std::memory_order_relaxed));
  }
}

#if 0
void DynamicBitSet::reset(size_t index) {
  size_t bit_index = index/bits_uint64;
  uint64_t bit_offset = 1;
  bit_offset <<= (index%bits_uint64);
  bitvec[bit_index].fetch_and(~bit_offset);
}
#endif

// assumes bit_vector is not updated (set) in parallel
void DynamicBitSet::bitwise_or(const DynamicBitSet& other) {
  assert(size() == other.size());
  auto& other_bitvec = other.get_vec();
  galois::do_all(galois::iterate(0ul, bitvec.size()),
                 [&](size_t i) {
                    bitvec[i] |= other_bitvec[i];
                 },
                 galois::no_stats());
}

uint64_t DynamicBitSet::count() {
  galois::GAccumulator<uint64_t> ret;
  galois::do_all(galois::iterate(bitvec.begin(), bitvec.end()),
                 [&](uint64_t n) {
#ifdef __GNUC__
                   ret += __builtin_popcountll(n);
#else
                   n = n - ((n >> 1) & 0x5555555555555555UL);
                   n = (n & 0x3333333333333333UL) + ((n >> 2) & 0x3333333333333333UL);
                   ret += (((n + (n >> 4)) & 0xF0F0F0F0F0F0F0FUL) * 0x101010101010101UL) >> 56;
#endif
                 },
                 galois::no_stats());
  return ret.reduce();
}

}
