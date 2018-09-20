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

/**
 * @file galois/DynamicBitset.h
 *
 * Contains the DynamicBitSet class and most of its implementation.
 */

#ifndef _GALOIS_DYNAMIC_BIT_SET_
#define _GALOIS_DYNAMIC_BIT_SET_

#include "galois/AtomicWrapper.h"
#include <boost/mpl/has_xxx.hpp>
#include <climits> // CHAR_BIT
#include <vector>
#include <assert.h>

namespace galois {
/**
 * Concurrent dynamically allocated bitset
 **/
class DynamicBitSet {
  std::vector<galois::CopyableAtomic<uint64_t>> bitvec;
  size_t num_bits;
  static constexpr uint32_t bits_uint64 = sizeof(uint64_t) * CHAR_BIT;

public:
  //! Constructor which initializes to an empty bitset.
  DynamicBitSet() : num_bits(0) {}

  /**
   * Returns the underlying bitset representation to the user
   *
   * @returns constant reference vector of copyable atomics that represents
   * the bitset
   */
  const std::vector<galois::CopyableAtomic<uint64_t>>& get_vec() const {
    return bitvec;
  }

  /**
   * Returns the underlying bitset representation to the user
   *
   * @returns reference to vector of copyable atomics that represents the
   * bitset
   */
  std::vector<galois::CopyableAtomic<uint64_t>>& get_vec() { return bitvec; }

  /**
   * Resizes the bitset.
   *
   * @param n Size to change the bitset to
   */
  void resize(uint64_t n) {
    assert(bits_uint64 == 64); // compatibility with other devices
    num_bits = n;
    bitvec.resize((n + bits_uint64 - 1) / bits_uint64);
  }

  /**
   * Gets the size of the bitset
   * @returns The number of bits held by the bitset
   */
  size_t size() const { return num_bits; }

  /**
   * Gets the space taken by the bitset
   * @returns the space in bytes taken by this bitset
   */
  size_t alloc_size() const { return bitvec.size() * sizeof(uint64_t); }

  /**
   * Unset every bit in the bitset.
   */
  void reset() { std::fill(bitvec.begin(), bitvec.end(), 0); }

  /**
   * Unset a range of bits given an inclusive range
   *
   * @param begin first bit in range to reset
   * @param end last bit in range to reset
   */
  void reset(size_t begin, size_t end);

  /**
   * Check a bit to see if it is currently set. Assumes the bit set is not
   * updated (set) in parallel.
   *
   * @param index Bit to check to see if set
   * @returns true if index is set
   */
  bool test(size_t index) const;

  /**
   * Set a bit in the bitset.
   *
   * @param index Bit to set
   */
  void set(size_t index);

#if 0
    void reset(size_t index);
#endif

  // assumes bit_vector is not updated (set) in parallel
  void bitwise_or(const DynamicBitSet& other);

  /**
   * Count how many bits are set in the bitset
   *
   * @returns number of set bits in the bitset
   */
  uint64_t count();

  //! this is defined to
  using tt_is_copyable = int;
};

//! An empty bitset object; used mainly by InvalidBitsetFnTy
static galois::DynamicBitSet EmptyBitset;

//! A structure representing an empty bitset.
struct InvalidBitsetFnTy {
  //! Returns false as this is an empty bitset
  static constexpr bool is_vector_bitset() { return false; }

  //! Returns false as this is an empty bitset (invalid)
  static constexpr bool is_valid() { return false; }

  //! Returns the empty bitset
  static galois::DynamicBitSet& get() { return EmptyBitset; }

  //! No-op since it's an empty bitset
  static void reset_range(size_t begin, size_t end) {}
};
} // namespace galois
#endif
