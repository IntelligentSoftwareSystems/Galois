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
      for(auto x = 0; x < std::ceil(n/bits_uint64); x++){
        bitvec[x] = Galois::CopyableAtomic<uint64_t>(0).load();
      }
    }

    const std::vector<Galois::CopyableAtomic<uint64_t>>& get_vec() const{
      return bitvec;
    }
 
    std::vector<Galois::CopyableAtomic<uint64_t>>& get_vec() {
      return bitvec;
    }

    void bit_set(uint32_t LID){
      assert(LID/bits_uint64 < bitvec.size());
      bitvec[LID/bits_uint64].fetch_or(1<<(LID%bits_uint64));
    }

    // single thread is accessing.
    void reset(){
      for(auto x : bitvec){
        x = 0;
      }
    }

    bool is_set(uint32_t index){
      assert(index/bits_uint64 < bitvec.size());
      return std::bitset<bits_uint64>(bitvec[index/bits_uint64])[index%bits_uint64];
#if 0
      if(std::bitset<bits_uint64>(bitvec[index/bits_uint64])[index%bits_uint64] == 0){
        return false;
      }else {
        return true;
      }
#endif
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
      uint32_t i = 0;
      for(auto j = 0;  (num - i) >= bits_uint64; i+=64, ++j){
        bits_set += std::bitset<bits_uint64>(bitvec[j]).count();
      }

      if(num - i > 0)
        bits_set += std::bitset<bits_uint64>(bitvec[i/bits_uint64] << (bits_uint64 - (num - i))).count();

      //for(auto x = i; x < num; ++x){
        //bits_set += is_set(x);
      //}

      return bits_set;
    }

    uint64_t size(){
      return bitvec.size();
    }
    typedef int tt_is_copyable;
  };
}
#endif
