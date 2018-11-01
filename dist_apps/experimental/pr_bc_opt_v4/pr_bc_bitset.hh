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
#ifndef _PR_BC_BITSET_
#define _PR_BC_BITSET_

#include "galois/AtomicWrapper.h"
#include "galois/PODResizeableArray.h"
#include <boost/random/detail/integer_log2.hpp>
#include <climits> // CHAR_BIT
#include <vector>

/**
 * Optimized mode: enable ONLY ONE of them at most
 */
// #define REVERSE_MODE
// #define FLIP_MODE


/**
 * Do you need an indicator?
 */
#define USE_INDICATOR

/**
 * Derivate from DynamicBitSet
 **/
// template <typename Block=uint32_t, typename Allocator=galois::gstl::Pow2Alloc<Block>>
class PRBCBitSet : public galois::DynamicBitSet {
  // @DynamicBitSet (protected)
  // galois::PODResizeableArray<galois::CopyableAtomic<uint64_t>> bitvec;
  // size_t num_bits;
  // static constexpr uint32_t bits_uint64 = sizeof(uint64_t) * CHAR_BIT;

  #ifdef USE_INDICATOR
    //! indicate the index of bit to process
    size_t indicator;
  #endif

  // Member functions
  inline size_t get_word(size_t pos) const { return pos < bits_uint64? 0 : pos / bits_uint64; }
  inline size_t get_offset(size_t pos) const { return pos < bits_uint64? pos : pos % bits_uint64; }
  inline uint64_t get_mask(size_t pos) const { return uint64_t(1) << get_offset(pos); }

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
      uint64_t block = get_word(pos), mask = get_mask(pos);
      uint64_t rBlock = get_word(reverse(pos)), rMask = get_mask(reverse(pos));
      // flip if asymmetrical
      if (!(bitvec[rBlock] & rMask)) {
        bitvec[block].fetch_and(~mask);
        size_t r_old = bitvec[rBlock];
        while (!bitvec[rBlock].compare_exchange_weak(
          r_old, r_old | rMask, std::memory_order_relaxed));
      }
    }
  #endif

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

public:
  //! sign for N/A
  static const size_t npos = std::numeric_limits<size_t>::max();

  #ifdef FLIP_MODE
    void flip() {
      size_t first = find_first();
      if (first != npos)
        flip_recursive(first);
    }
  #endif

  #ifdef USE_INDICATOR
  // Accessors
  size_t getIndicator() const { return indicator; }
  void setIndicator(size_t index) { indicator = index; }
  #endif

  // @DynamicBitSet
  size_t size() const { return num_bits; }

  //! Constructor which initializes to an empty bitset.
  PRBCBitSet() {
    num_bits = numSourcesPerRound;
    bitvec.resize((numSourcesPerRound - 1) / bits_uint64 + 1);
    reset();
    #ifdef USE_INDICATOR
      indicator = npos;
    #endif
  }

  // assumes bit_vector is not updated (set) in parallel
  bool test(size_t index) const {
    size_t bit_index = get_word(index);
    uint64_t bit_mask = get_mask(index);
    return ((bitvec[bit_index] & bit_mask) != 0);
  }

  /**
   * Test and then set
   *
   * @returns test result before set
   */
  bool test_set(size_t pos, bool val=true) {
    bool const ret = test(pos);
    if (ret != val) {
      uint64_t old_val = bitvec[get_word(pos)];
      if (val) {
        while (!bitvec[get_word(pos)].compare_exchange_weak(
          old_val, (old_val | get_mask(pos)), 
          std::memory_order_relaxed));
      }
      else {
        while (!bitvec[get_word(pos)].compare_exchange_weak(
          old_val, (old_val & ~get_mask(pos)), 
          std::memory_order_relaxed));
      }
    }
    return ret;
  }

  void set(size_t index) {
    size_t bit_index = get_word(index);
    uint64_t bit_mask = get_mask(index);
    if ((bitvec[bit_index] & bit_mask) == 0) { // test and set
      size_t old_val = bitvec[bit_index];
      while (!bitvec[bit_index].compare_exchange_weak(
          old_val, old_val | bit_mask, 
          std::memory_order_relaxed));
    }
  }

  void reset(size_t index) {
    size_t bit_index = get_word(index);
    uint64_t bit_mask = get_mask(index);
    bitvec[bit_index].fetch_and(~bit_mask);
    // @ Better implementation:
    // while (!bitvec[bit_index].compare_exchange_weak(
    //   old_val, old_val & ~bit_mask, 
    //   std::memory_order_relaxed));
  }

  // @DynamicBitSet
  void reset() { std::fill(bitvec.begin(), bitvec.end(), uint64_t(0)); }

  // @DynamicBitSet
  void resize(uint64_t n) {
    assert(bits_uint64 == 64); // compatibility with other devices
    num_bits = n;
    bitvec.resize((n + bits_uint64 - 1) / bits_uint64);
    reset();
  }

  bool none() {
    for (size_t i = 0; i < bitvec.size(); ++i)
      if (bitvec[i])
        return false;
    return true;
  }

  #ifdef USE_INDICATOR
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

    bool test_set_indicator(size_t pos, bool val=true) {
      #ifdef REVERSE_MODE
        if (test_set(reverse(pos), val)) {
          if (pos == indicator) {
            forward_indicator();
          }
          return true;
        }
        else
          return false;
      #else
        if (test_set(pos, val)) {
          if (pos == indicator) {
            forward_indicator();
          }
          return true;
        }
        else
          return false;
      #endif
    }

    /**
     * Return true if indicator is npos
     */
    bool nposInd() {
      return indicator == npos;
    }
  #endif
  /**
   * Returns: the lowest index i such as bit i is set, or npos if *this has no on bits.
   */
  size_t find_first() const {
    return find_from_block(0);
  }

  size_t find_last() const {
    return find_from_block(bitvec.size() - 1, false);
  }

  inline size_t begin() { return find_first(); }
  inline size_t rbegin() { return find_last(); }
  inline size_t end() { return npos; }
  inline size_t rend() { return npos; }

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
    size_t curBlock = get_word(pos);
    uint64_t res = bitvec[curBlock] >> get_offset(pos);
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
    size_t curBlock = get_word(pos);
    uint64_t res = bitvec[curBlock] & ((uint64_t(2) << get_offset(pos)) - 1);
    return res?
      curBlock * bits_uint64 + left_most_bit(res) : 
      (curBlock?
        find_from_block(--curBlock, false) : npos);
  }

  #ifdef USE_INDICATOR
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
#endif