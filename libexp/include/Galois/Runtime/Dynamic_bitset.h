//Dynamic bit set for CPU

#include <atomic>
#include <iostream>
#include <bitset>
#include "Galois/Atomic_wrapper.h"
#include <climits> // CHAR_BIT

#ifndef _GALOIS_DYNAMIC_BIT_SET_
#define _GALOIS_DYNAMIC_BIT_SET_

namespace Galois {

  class DynamicBitSet {

    std::vector<Galois::CopyableAtomic<uint64_t>> bitvec;
    static constexpr uint32_t bits_uint64 = sizeof(uint64_t)*CHAR_BIT;

  public:

    void resize_init(uint64_t n){
      bitvec.resize(std::ceil((float)n/bits_uint64));
      for(uint32_t x = 0; x < bitvec.size(); x++){
        bitvec[x] = Galois::CopyableAtomic<uint64_t>(0).load();
      }
    }

    const std::vector<Galois::CopyableAtomic<uint64_t>>& get_vec() const{
      return bitvec;
    }
 
    std::vector<Galois::CopyableAtomic<uint64_t>>& get_vec() {
      return bitvec;
    }

    void bit_set(uint32_t index){
      uint32_t bit_index = index/64;
      assert(bit_index < bitvec.size());
      uint64_t bit_offset = 1;
      bit_offset <<= (index%64);
      bitvec[bit_index].fetch_or(bit_offset);
    }

    // single thread is accessing.
    void reset(){
      for(auto x : bitvec){
        x = 0;
      }
    }

    bool is_set(uint32_t index){
      uint32_t bit_index = index/64;
      assert(bit_index < bitvec.size());
      uint64_t bit_offset = 1;
      bit_offset <<= (index%64);
      return ((bitvec[bit_index] & bit_offset) != 0);
    }

    uint64_t bit_count(){
      uint64_t bits_set = 0;
      for(auto x : bitvec){
        bits_set += std::bitset<bits_uint64>(x).count();
      }
      return bits_set;
    }

    uint64_t bit_count(uint32_t num){
      uint64_t bits_set = 0;
      unsigned index = num/bits_uint64;
      unsigned offset = num%bits_uint64;
      for (unsigned i = 0; i < index; ++i) {
        bits_set += std::bitset<bits_uint64>(bitvec[i]).count();
      }
      if (offset > 0) {
        unsigned long long int value = bitvec[index];
        value <<= (bits_uint64-offset);
        bits_set += std::bitset<bits_uint64>(value).count();
      }
      return bits_set;
    }

    uint64_t size(){
      return bitvec.size();
    }
    typedef int tt_is_copyable;
  };
}
#endif
