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
    }

    size_t size() const {
      return num_bits;
    }

    size_t alloc_size() const {
      return bitvec.size() * sizeof(uint64_t);
    }

    void clear(){
      std::fill(bitvec.begin(), bitvec.end(), 0);
    }

    // assumes bit_vector is not updated (set) in parallel
    bool test(uint32_t index) const {
      uint32_t bit_index = index/bits_uint64;
      uint64_t bit_offset = 1;
      bit_offset <<= (index%bits_uint64);
      return ((bitvec[bit_index] & bit_offset) != 0);
    }

    void set(uint32_t index){
      uint32_t bit_index = index/bits_uint64;
      uint64_t bit_offset = 1;
      bit_offset <<= (index%bits_uint64);
      bitvec[bit_index].fetch_or(bit_offset);
    }

#if 0
    void reset(uint32_t index){
      uint32_t bit_index = index/bits_uint64;
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
} // namespace Galois
#endif
