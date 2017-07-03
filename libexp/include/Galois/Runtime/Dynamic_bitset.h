//Dynamic bit set for CPU

#include <atomic>
#include <iostream>
#include <bitset>
#include "Galois/Atomic_wrapper.h"
#include <climits> // CHAR_BIT
#include <boost/iterator/counting_iterator.hpp>
#include "Galois/GaloisForwardDecl.h"
#include "Galois/Traits.h"

#ifndef _GALOIS_DYNAMIC_BIT_SET_
#define _GALOIS_DYNAMIC_BIT_SET_

namespace Galois {

  class DynamicBitSet {

    std::vector<Galois::CopyableAtomic<uint64_t>> bitvec;
    size_t num_bits;
    static constexpr uint32_t bits_uint64 = sizeof(uint64_t)*CHAR_BIT;

  public:

    DynamicBitSet() : num_bits(0) {}

    const std::vector<Galois::CopyableAtomic<uint64_t>>& get_vec() const{
      return bitvec;
    }
 
    std::vector<Galois::CopyableAtomic<uint64_t>>& get_vec() {
      return bitvec;
    }

    void resize(uint64_t n){
      assert(bits_uint64 == 64); // compatibility with other devices
      num_bits = n;
      bitvec.resize(std::ceil((float)n/bits_uint64));
      reset();
    }

    size_t size() const {
      return num_bits;
    }

    size_t alloc_size() const {
      return bitvec.size() * sizeof(uint64_t);
    }

    void reset(){
      std::fill(bitvec.begin(), bitvec.end(), 0);
    }

    // inclusive range
    void reset(size_t begin, size_t end){
      assert(begin <= (num_bits - 1));
      assert(end <= (num_bits - 1));
      size_t vec_begin = ceil((float)begin/bits_uint64);
      size_t vec_end;
      if (end == (num_bits - 1)) vec_end = bitvec.size();
      else vec_end = (end+1)/bits_uint64; // floor
      if (vec_begin < vec_end) {
        std::fill(bitvec.begin() + vec_begin, bitvec.begin() + vec_end, 0);
      }
      vec_begin *= bits_uint64;
      vec_end *= bits_uint64;
      if (begin < vec_begin) {
        size_t diff = vec_begin - begin;
        assert(diff < 64);
        uint64_t mask = (1 << (64 - diff)) - 1;
        size_t bit_index = begin/bits_uint64;
        bitvec[bit_index] &= mask;
      }
      if (end >= vec_end) {
        size_t diff = end - vec_end + 1;
        assert(diff < 64);
        uint64_t mask = (1 << diff) - 1;
        size_t bit_index = end/bits_uint64;
        bitvec[bit_index] &= ~mask;
      }
    }

    // assumes bit_vector is not updated (set) in parallel
    bool test(size_t index) const {
      size_t bit_index = index/bits_uint64;
      uint64_t bit_offset = 1;
      bit_offset <<= (index%bits_uint64);
      return ((bitvec[bit_index] & bit_offset) != 0);
    }

    void set(size_t index){
      size_t bit_index = index/bits_uint64;
      uint64_t bit_offset = 1;
      bit_offset <<= (index%bits_uint64);
      bitvec[bit_index].fetch_or(bit_offset);
    }

#if 0
    void reset(size_t index){
      size_t bit_index = index/bits_uint64;
      uint64_t bit_offset = 1;
      bit_offset <<= (index%bits_uint64);
      bitvec[bit_index].fetch_and(~bit_offset);
    }
#endif

    // assumes bit_vector is not updated (set) in parallel
    void bitwise_or(const DynamicBitSet& other) {
      assert(size() == other.size());
      auto& other_bitvec = other.get_vec();
      for (size_t i = 0; i < bitvec.size(); ++i) {
        bitvec[i] |= other_bitvec[i];
      }
    }

    typedef int tt_is_copyable;
  };

  static Galois::DynamicBitSet EmptyBitset;

  struct InvalidBitsetFnTy {
    static bool is_valid() {
      return false;
    }
    static Galois::DynamicBitSet& get() {
      return EmptyBitset;
    }
    // inclusive range
    static void reset_range(size_t begin, size_t end) {
    }
  };
} // namespace Galois
#endif
