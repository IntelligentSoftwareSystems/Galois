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
#include "galois/PODResizeableArray.h"
#include <boost/mpl/has_xxx.hpp>
#include <climits> // CHAR_BIT
#include <vector>
#include <assert.h>

#ifdef _PR_BC_OPT_V3_

#include <boost/random/detail/integer_log2.hpp>

/**
 * Optimized mode: enable ONLY ONE of them at most
 */
// #define REVERSE_MODE
#define FLIP_MODE

#endif

namespace galois {
/**
 * Concurrent dynamically allocated bitset
 **/
class DynamicBitSet {
  galois::PODResizeableArray<galois::CopyableAtomic<uint64_t>> bitvec;
  size_t num_bits;
  static constexpr uint32_t bits_uint64 = sizeof(uint64_t) * CHAR_BIT;

#ifdef _PR_BC_OPT_V3_
  //! indicate the index of bit to process
  size_t indicator;

  // Member functions
  size_t block_index(size_t pos) const { return pos < bits_uint64? 0 : pos / bits_uint64; }
  size_t bit_index(size_t pos) const { return pos < bits_uint64? pos : pos % bits_uint64; }
  uint64_t bit_mask(size_t pos) const { return uint64_t(1) << bit_index(pos); }

  #if defined(REVERSE_MODE) || defined(FLIP_MODE)
    size_t reverse(size_t pos) {
      return pos == npos? npos : num_bits - pos - 1;
    }
  #endif

  #ifdef FLIP_MODE
    void flip_recursive(size_t pos) {
      size_t next = find_next(pos);
      if (next != npos)
        flip_recursive(next);
      // do the flip for pos
      uint64_t block = block_index(pos), mask = bit_mask(pos);
      uint64_t rBlock = block_index(reverse(pos)), rMask = bit_mask(reverse(pos));
      // flip if asymmetrical
      if (!(bitvec[rBlock] & rMask)) {
        bitvec[block].fetch_and(~mask);
        size_t r_old = bitvec[rBlock];
        while (!bitvec[rBlock].compare_exchange_weak(
          r_old, r_old | rMask, std::memory_order_relaxed));
      }
    }
  #endif

#endif

public:
  //! Constructor which initializes to an empty bitset.
  DynamicBitSet() : num_bits(0) {
  #ifdef _PR_BC_OPT_V3_
    indicator = npos;
  #endif
  }

  /**
   * Returns the underlying bitset representation to the user
   *
   * @returns constant reference vector of copyable atomics that represents
   * the bitset
   */
  const auto& get_vec() const {
    return bitvec;
  }

  /**
   * Returns the underlying bitset representation to the user
   *
   * @returns reference to vector of copyable atomics that represents the
   * bitset
   */
  auto& get_vec() { return bitvec; }

  /**
   * Resizes the bitset.
   *
   * @param n Size to change the bitset to
   */
  void resize(uint64_t n) {
    assert(bits_uint64 == 64); // compatibility with other devices
    num_bits = n;
    bitvec.resize((n + bits_uint64 - 1) / bits_uint64);
    reset();
  }

  /**
   * Reserves capacity for the bitset.
   *
   * @param n Size to reserve the capacity of the bitset to 
   */
  void reserve(uint64_t n) {
    assert(bits_uint64 == 64); // compatibility with other devices
    bitvec.reserve((n + bits_uint64 - 1) / bits_uint64);
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
  //size_t alloc_size() const { return bitvec.size() * sizeof(uint64_t); }

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

#if 0 || defined(_PR_BC_OPT_V3_)
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

  /**
   * Returns a vector containing the set bits in this bitset in order
   * from left to right.
   * Do NOT call in a parallel region as it uses galois::on_each.
   *
   * @returns vector with offsets into set bits
   */
  std::vector<uint32_t> getOffsets();

  //! this is defined to
  using tt_is_copyable = int;

#ifdef _PR_BC_OPT_V3_
  //! not found
  static const size_t npos = std::numeric_limits<size_t>::max();

  #ifdef FLIP_MODE
    void flip() {
      size_t first = find_first();
      if (first != npos)
        flip_recursive(first);
    }
  #endif

  // Accessors
  size_t getIndicator() const { return indicator; }
  void setIndicator(size_t index) { indicator = index; }

  /**
   * Test and then set
   *
   * @returns test result before set
   */
  bool test_set(size_t pos, bool val=true) {
    bool const ret = test(pos);
    if (ret != val) {
      uint64_t old_val = bitvec[block_index(pos)];
      if (val) {
        while (!bitvec[block_index(pos)].compare_exchange_weak(
          old_val, (old_val | bit_mask(pos)), 
          std::memory_order_relaxed));
      }
      else {
        while (!bitvec[block_index(pos)].compare_exchange_weak(
          old_val, (old_val & ~bit_mask(pos)), 
          std::memory_order_relaxed));
      }
    }
    return ret;
  }

  bool none() {
    for (size_t i = 0; i < bitvec.size(); ++i)
      if (bitvec[i])
        return false;
    return true;
  }
    
  /**
   * Set a bit with the side-effect updating indicator to the first.
   */
  void set_indicator(size_t pos) {
    #ifdef REVERSE_MODE
      set(reverse(pos));
    #else
      set(pos);
    #endif
    if (pos < indicator) {
      indicator = pos;
    }
  }

  void test_set_indicator(size_t pos, bool val=true) {
    #ifdef REVERSE_MODE
      if (test_set(reverse(pos), val)) {
        if (pos == indicator) {
          forward_indicator();
        }
      }
    #else
      if (test_set(pos, val)) {
        if (pos == indicator) {
          forward_indicator();
        }
      }
    #endif
  }

  /**
   * Return true if indicator is npos
   */
  bool nposInd() {
    return indicator == npos;
  }

  size_t right_most_bit(uint64_t w) const {
    // assert(w >= 1);
    return boost::integer_log2<uint64_t>(w & -w);
  }

  size_t left_most_bit(uint64_t w) const {
      return boost::integer_log2<uint64_t>(w);
  }

size_t find_from_block(size_t first, bool fore=true) const {
  size_t i;
  if (fore) {
    for (i = first; i < bitvec.size() && bitvec[i] == 0; i++);
    if (i >= bitvec.size())
        return npos;
    return i * bits_uint64 + right_most_bit(bitvec[i]);
  }
  else {
    for (i = first; i > 0 && bitvec[i] == 0; i--);
    if (i <= 0 && bitvec[i] == 0)
      return npos;
    return i * bits_uint64 + left_most_bit(bitvec[i]);
  }
}

/**
 * Returns: the lowest index i such as bit i is set, or npos if *this has no on bits.
 */
size_t find_first() const {
  return find_from_block(0);
}

size_t find_last() const {
  return find_from_block(bitvec.size() - 1, false);
}

/**
 * Returns: the lowest index i greater than pos such as bit i is set, or npos if no such index exists.
 */
size_t find_next(size_t pos) const {
  if (pos == npos) {
    return find_first();
  }
  if (++pos >= size() || size() == 0) {
    return npos;
  }
  size_t curBlock = block_index(pos);
  uint64_t res = bitvec[curBlock] >> bit_index(pos);
  return res?
    pos + right_most_bit(res) : find_from_block(++curBlock);
}

size_t find_prev(size_t pos) const{
  if (pos >= size()) {
    return find_last();
  }
  // Return npos if no bit set
  if (pos-- == 0 || size() == 0) {
    return npos;
  }
  size_t curBlock = block_index(pos);
  uint64_t res = bitvec[curBlock] & ((uint64_t(2) << bit_index(pos)) - 1);
  return res?
    curBlock * bits_uint64 + left_most_bit(res) : 
    (curBlock?
      find_from_block(--curBlock, false) : npos);
}

/**
 * To move indicator to the previous set bit, and return the old value.
 */
size_t forward_indicator() {
  size_t old = indicator;
  #ifdef REVERSE_MODE
    indicator = reverse(find_prev(reverse(indicator)));
  #else
    indicator = find_next(indicator);
  #endif
  return old;
}

/**
 * To move indicator to the next set bit.
 */
size_t backward_indicator() {
  size_t old = indicator;
  #ifdef FLIP_MODE
    indicator = nposInd()? find_first() : find_next(indicator);
    return reverse(old);
  #else
    #ifdef REVERSE_MODE
    indicator = reverse(find_next(reverse(indicator)));
    #else
    indicator = find_prev(indicator);
    #endif
    return old;
  #endif
}
#endif
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
