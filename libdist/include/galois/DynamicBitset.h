/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#ifndef _GALOIS_DYNAMIC_BIT_SET_
#define _GALOIS_DYNAMIC_BIT_SET_

#include "galois/AtomicWrapper.h"
#include <boost/mpl/has_xxx.hpp>
#include <climits> // CHAR_BIT
#include <vector>
#include <assert.h>

namespace galois {
  /* Concurrent dynamically allocated bitset */
  class DynamicBitSet {
    std::vector<galois::CopyableAtomic<uint64_t>> bitvec;
    size_t num_bits;
    static constexpr uint32_t bits_uint64 = sizeof(uint64_t) * CHAR_BIT;

  public:
    DynamicBitSet() : num_bits(0) {}

    const std::vector<galois::CopyableAtomic<uint64_t>>& get_vec() const {
      return bitvec;
    }
 
    std::vector<galois::CopyableAtomic<uint64_t>>& get_vec() {
      return bitvec;
    }

    void resize(uint64_t n) {
      assert(bits_uint64 == 64); // compatibility with other devices
      num_bits = n;
      bitvec.resize((n + bits_uint64 - 1)/bits_uint64);
      reset();
    }

    size_t size() const {
      return num_bits;
    }

    size_t alloc_size() const {
      return bitvec.size() * sizeof(uint64_t);
    }

    void reset() {
      std::fill(bitvec.begin(), bitvec.end(), 0);
    }

    // inclusive range
    void reset(size_t begin, size_t end);

    // assumes bit_vector is not updated (set) in parallel
    bool test(size_t index) const;

    void set(size_t index);

#if 0
    void reset(size_t index);
#endif

    // assumes bit_vector is not updated (set) in parallel
    void bitwise_or(const DynamicBitSet& other);

    uint64_t count();

    typedef int tt_is_copyable;
  };

  static galois::DynamicBitSet EmptyBitset;

  struct InvalidBitsetFnTy {
    static constexpr bool is_vector_bitset() {
      return false;
    }

    static constexpr bool is_valid() {
      return false;
    }

    static galois::DynamicBitSet& get() {
      return EmptyBitset;
    }

    static void reset_range(size_t begin, size_t end) {
    }
  };
} // namespace galois
#endif
