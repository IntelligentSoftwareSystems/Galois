/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

/*
 */

/**
 * @file cuda/DynamicBitset.h
 *
 * Contains implementation of CUDA dynamic bitset and iterators for it.
 */

// thread-safe dynamic bitset in CUDA
#pragma once
#include <cuda.h>
#include <math.h>
#include <iterator>

/**
 * Dynamic Bitset, CUDA version. See galois/DynamicBitset.h.
 *
 * @todo document this file
 */
class DynamicBitset {
  size_t num_bits_capacity;
  size_t num_bits;
  uint64_t* bit_vector;

public:
  DynamicBitset() {
    num_bits_capacity = 0;
    num_bits          = 0;
    bit_vector        = NULL;
  }

  DynamicBitset(size_t nbits) { alloc(nbits); }

  ~DynamicBitset() {
    if (bit_vector != NULL)
      cudaFree(bit_vector);
  }

  void alloc(size_t nbits) {
    assert(num_bits == 0);
    assert(sizeof(unsigned long long int) * 8 == 64);
    assert(sizeof(uint64_t) * 8 == 64);
    num_bits_capacity = nbits;
    num_bits          = nbits;
    CUDA_SAFE_CALL(cudaMalloc(&bit_vector, vec_size() * sizeof(uint64_t)));
    reset();
  }

  void resize(size_t nbits) {
    assert(nbits <= num_bits_capacity);
    num_bits = nbits;
  }

  __device__ __host__ size_t size() const { return num_bits; }

  __device__ __host__ size_t vec_size() const {
    size_t bit_vector_size = (num_bits + 63) / 64;
    return bit_vector_size;
  }

  __device__ __host__ size_t alloc_size() const {
    return vec_size() * sizeof(uint64_t);
  }

  void reset() {
    CUDA_SAFE_CALL(cudaMemset(bit_vector, 0, vec_size() * sizeof(uint64_t)));
  }

  // assumes bit_vector is not updated (set) in parallel
  __device__ bool test(const size_t id) const {
    size_t bit_index    = id / 64;
    uint64_t bit_offset = 1;
    bit_offset <<= (id % 64);
    return ((bit_vector[bit_index] & bit_offset) != 0);
  }

  __device__ void set(const size_t id) {
    size_t bit_index                  = id / 64;
    unsigned long long int bit_offset = 1;
    bit_offset <<= (id % 64);
    if ((bit_vector[bit_index] & bit_offset) == 0) { // test and set
      atomicOr((unsigned long long int*)&bit_vector[bit_index], bit_offset);
    }
  }

  // different indices can be updated in parallel
  __device__ void batch_reset(const size_t bit_index) {
    bit_vector[bit_index] = 0;
  }

  // different indices can be updated in parallel
  // but assumes same index is not updated in parallel
  __device__ void batch_bitwise_and(const size_t bit_index,
                                    const uint64_t mask) {
    bit_vector[bit_index] &= mask;
  }

  void copy_to_cpu(uint64_t* bit_vector_cpu_copy) {
    assert(bit_vector_cpu_copy != NULL);
    CUDA_SAFE_CALL(cudaMemcpy(bit_vector_cpu_copy, bit_vector,
                              vec_size() * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
  }

  void copy_to_gpu(uint64_t* cpu_bit_vector) {
    assert(cpu_bit_vector != NULL);
    CUDA_SAFE_CALL(cudaMemcpy(bit_vector, cpu_bit_vector,
                              vec_size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));
  }
};

class DynamicBitsetIterator
    : public std::iterator<std::random_access_iterator_tag, bool> {
  DynamicBitset* bitset;
  size_t offset;

public:
  __device__ __host__ __forceinline__ DynamicBitsetIterator(DynamicBitset* b,
                                                            size_t i = 0)
      : bitset(b), offset(i) {}

  __device__ __host__ __forceinline__ DynamicBitsetIterator& operator++() {
    offset++;
    return *this;
  }

  __device__ __host__ __forceinline__ DynamicBitsetIterator& operator--() {
    offset--;
    return *this;
  }

  __device__ __host__ __forceinline__ bool
  operator<(const DynamicBitsetIterator& bi) {
    return (offset < bi.offset);
  }

  __device__ __host__ __forceinline__ bool
  operator<=(const DynamicBitsetIterator& bi) {
    return (offset <= bi.offset);
  }

  __device__ __host__ __forceinline__ bool
  operator>(const DynamicBitsetIterator& bi) {
    return (offset > bi.offset);
  }

  __device__ __host__ __forceinline__ bool
  operator>=(const DynamicBitsetIterator& bi) {
    return (offset >= bi.offset);
  }

  __device__ __host__ __forceinline__ DynamicBitsetIterator&
  operator+=(size_t i) {
    offset += i;
    return *this;
  }

  __device__ __host__ __forceinline__ DynamicBitsetIterator&
  operator-=(size_t i) {
    offset -= i;
    return *this;
  }

  __device__ __host__ __forceinline__ DynamicBitsetIterator
  operator+(size_t i) {
    return DynamicBitsetIterator(bitset, offset + i);
  }

  __device__ __host__ __forceinline__ DynamicBitsetIterator
  operator-(size_t i) {
    return DynamicBitsetIterator(bitset, offset - i);
  }

  __device__ __host__ __forceinline__ difference_type
  operator-(const DynamicBitsetIterator& bi) {
    return (offset - bi.offset);
  }

  __device__ __forceinline__ bool operator*() const {
    return bitset->test(offset);
  }

  __device__ __forceinline__ bool operator[](const size_t id) const {
    return bitset->test(offset + id);
  }
};

class IdentityIterator
    : public std::iterator<std::random_access_iterator_tag, size_t> {
  size_t offset;

public:
  __device__ __host__ __forceinline__ IdentityIterator(size_t i = 0)
      : offset(i) {}

  __device__ __host__ __forceinline__ IdentityIterator& operator++() {
    offset++;
    return *this;
  }

  __device__ __host__ __forceinline__ IdentityIterator& operator--() {
    offset--;
    return *this;
  }

  __device__ __host__ __forceinline__ bool
  operator<(const IdentityIterator& bi) {
    return (offset < bi.offset);
  }

  __device__ __host__ __forceinline__ bool
  operator<=(const IdentityIterator& bi) {
    return (offset <= bi.offset);
  }

  __device__ __host__ __forceinline__ bool
  operator>(const IdentityIterator& bi) {
    return (offset > bi.offset);
  }

  __device__ __host__ __forceinline__ bool
  operator>=(const IdentityIterator& bi) {
    return (offset >= bi.offset);
  }

  __device__ __host__ __forceinline__ IdentityIterator& operator+=(size_t i) {
    offset += i;
    return *this;
  }

  __device__ __host__ __forceinline__ IdentityIterator& operator-=(size_t i) {
    offset -= i;
    return *this;
  }

  __device__ __host__ __forceinline__ IdentityIterator operator+(size_t i) {
    return IdentityIterator(offset + i);
  }

  __device__ __host__ __forceinline__ IdentityIterator operator-(size_t i) {
    return IdentityIterator(offset - i);
  }

  __device__ __host__ __forceinline__ difference_type
  operator-(const IdentityIterator& bi) {
    return (offset - bi.offset);
  }

  __device__ __forceinline__ size_t operator*() const { return offset; }

  __device__ __forceinline__ size_t operator[](const size_t id) const {
    return offset + id;
  }
};
