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
#include "galois/GaloisForwardDecl.h"
#include "galois/Traits.h"
#include "galois/Galois.h"
#include <boost/iterator/counting_iterator.hpp>
#include <boost/mpl/has_xxx.hpp>
#include <climits> // CHAR_BIT
#include <vector>
#include <assert.h>

namespace galois {
/**
 * Concurrent dynamically allocated bitset
 **/
template <typename _Tp=galois::CopyableAtomic<uint64_t>, 
          typename _Alloc=std::allocator<_Tp>, 
          typename _VecTp=galois::PODResizeableArray<_Tp, _Alloc>>
class DynamicBitSet {
protected:
  _VecTp bitvec;
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
  void reset(size_t begin, size_t end) {
    if (num_bits == 0) return;

    assert(begin <= (num_bits - 1));
    assert(end <= (num_bits - 1));

    // 100% safe implementation, but slow
    // for (unsigned long i = begin; i <= end; i++) {
    //  size_t bit_index = i / bits_uint64;
    //  uint64_t bit_offset = 1;
    //  bit_offset <<= (i % bits_uint64);
    //  uint64_t mask = ~bit_offset;
    //  bitvec[bit_index] &= mask;
    //}

    // block which you are safe to clear
    size_t vec_begin = (begin + bits_uint64 - 1) / bits_uint64;
    size_t vec_end;

    if (end == (num_bits - 1))
      vec_end = bitvec.size();
    else
      vec_end = (end + 1) / bits_uint64; // floor

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

        size_t end_diff  = end - vec_end + 1;
        uint64_t or_mask = ((uint64_t)1 << end_diff) - 1;
        mask |= ~or_mask;

        size_t bit_index = begin / bits_uint64;
        bitvec[bit_index] &= mask;
      }
    } else {
      if (begin < vec_begin) {
        size_t diff = vec_begin - begin;
        assert(diff < 64);
        uint64_t mask    = ((uint64_t)1 << (64 - diff)) - 1;
        size_t bit_index = begin / bits_uint64;
        bitvec[bit_index] &= mask;
      }
      if (end >= vec_end) {
        size_t diff = end - vec_end + 1;
        assert(diff < 64);
        uint64_t mask    = ((uint64_t)1 << diff) - 1;
        size_t bit_index = end / bits_uint64;
        bitvec[bit_index] &= ~mask;
      }
    }
  }

  /**
   * Check a bit to see if it is currently set. Assumes the bit set is not
   * updated (set) in parallel.
   *
   * @param index Bit to check to see if set
   * @returns true if index is set
   */
  bool test(size_t index)  const {
    size_t bit_index    = index / bits_uint64;
    uint64_t bit_offset = 1;
    bit_offset <<= (index % bits_uint64);
    return ((bitvec[bit_index] & bit_offset) != 0);
  }

  /**
   * Set a bit in the bitset.
   *
   * @param index Bit to set
   */
  void set(size_t index) {
    size_t bit_index    = index / bits_uint64;
    uint64_t bit_offset = 1;
    bit_offset <<= (index % bits_uint64);
    if ((bitvec[bit_index] & bit_offset) == 0) { // test and set
      size_t old_val = bitvec[bit_index];
      while (!bitvec[bit_index].compare_exchange_weak(
          old_val, old_val | bit_offset, std::memory_order_relaxed))
        ;
    }
  }

#if 0
    void reset(size_t index) {
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
    galois::do_all(galois::iterate(0ul, bitvec.size()),
                   [&](size_t i) { bitvec[i] |= other_bitvec[i]; },
                   galois::no_stats());
  }

  /**
   * Count how many bits are set in the bitset
   *
   * @returns number of set bits in the bitset
   */
  uint64_t count() {
    galois::GAccumulator<uint64_t> ret;
    galois::do_all(galois::iterate(bitvec.begin(), bitvec.end()),
                   [&](uint64_t n) {
  #ifdef __GNUC__
                     ret += __builtin_popcountll(n);
  #else
                     n = n - ((n >> 1) & 0x5555555555555555UL);
                     n = (n & 0x3333333333333333UL) +
                         ((n >> 2) & 0x3333333333333333UL);
                     ret += (((n + (n >> 4)) & 0xF0F0F0F0F0F0F0FUL) *
                             0x101010101010101UL) >>
                            56;
  #endif
                   },
                   galois::no_stats());
    return ret.reduce();
  }

  /**
   * Returns a vector containing the set bits in this bitset in order
   * from left to right.
   * Do NOT call in a parallel region as it uses galois::on_each.
   *
   * @returns vector with offsets into set bits
   */
  std::vector<uint32_t> getOffsets() {
    uint32_t activeThreads = galois::getActiveThreads();
    std::vector<unsigned int> tPrefixBitCounts(activeThreads);

    // count how many bits are set on each thread
    galois::on_each(
      [&] (unsigned tid, unsigned nthreads) {
        size_t start;
        size_t end;
        std::tie(start, end) = galois::block_range((size_t)0, this->size(), tid,
                                                   nthreads);

        unsigned int count = 0;
        for (unsigned int i = start; i < end; ++i) {
          if (this->test(i)) ++count;
        }

        tPrefixBitCounts[tid] = count;
    });

    // calculate prefix sum of bits per thread
    for (unsigned int i = 1; i < activeThreads; ++i) {
      tPrefixBitCounts[i] += tPrefixBitCounts[i - 1];
    }

    // total num of set bits
    uint64_t bitsetCount = tPrefixBitCounts[activeThreads - 1];
    std::vector<uint32_t> offsets;

    // calculate the indices of the set bits and save them to the offset
    // vector
    if (bitsetCount > 0) {
      offsets.resize(bitsetCount);
      galois::on_each(
        [&] (unsigned tid, unsigned nthreads) {
          size_t start;
          size_t end;
          std::tie(start, end) = galois::block_range((size_t)0, this->size(), tid,
                                                     nthreads);
          unsigned int count = 0;
          unsigned int tPrefixBitCount;
          if (tid == 0) {
            tPrefixBitCount = 0;
          } else {
            tPrefixBitCount = tPrefixBitCounts[tid - 1];
          }

          for (unsigned int i = start; i < end; ++i) {
            if (this->test(i)) {
              offsets[tPrefixBitCount + count] = i;
              ++count;
            }
          }
        }
      );
    }

    return offsets;
  }

  //! this is defined to
  using tt_is_copyable = int;
};

//! An empty bitset object; used mainly by InvalidBitsetFnTy
static galois::DynamicBitSet<> EmptyBitset;

//! A structure representing an empty bitset.
struct InvalidBitsetFnTy {
  //! Returns false as this is an empty bitset
  static constexpr bool is_vector_bitset() { return false; }

  //! Returns false as this is an empty bitset (invalid)
  static constexpr bool is_valid() { return false; }

  //! Returns the empty bitset
  static galois::DynamicBitSet<>& get() { return EmptyBitset; }

  //! No-op since it's an empty bitset
  static void reset_range(size_t begin, size_t end) {}
};
} // namespace galois
#endif
