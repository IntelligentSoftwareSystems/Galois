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

    void resize(uint64_t n){
      bitvec.resize(std::ceil((float)n/bits_uint64));
    }

    void init(){
      std::fill(bitvec.begin(), bitvec.end(), 0);
      /* auto init_ = [&](uint32_t x) { */
      /*   bitvec[x] = Galois::CopyableAtomic<uint64_t>(0).load(); */
      /* }; */
      //      Galois::do_all(boost::counting_iterator<uint64_t>(0), boost::counting_iterator<uint64_t>(bitvec.size()), init_, Galois::loopname("BITSET_INIT"));
    }

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
      Galois::do_all(boost::counting_iterator<uint64_t>(0), boost::counting_iterator<uint64_t>(bitvec.size()), [&](uint32_t x) {
        bitvec[x] = Galois::CopyableAtomic<uint64_t>(0).load();
      }, Galois::loopname("BITSET_CLEAR"));
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

    typedef int tt_is_copyable;
  };
} // namespace Galois
#endif
