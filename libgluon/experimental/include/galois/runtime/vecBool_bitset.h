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

namespace galois {

class VecBool {
  std::vector<uint8_t> vec;
  uint64_t numItems;
  uint32_t size_each;

public:
  VecBool() = default;
  VecBool(uint64_t _numItems, uint32_t _size_each)
      : numItems(_numItems), size_each(_size_each) {
    vec.resize((_numItems * _size_each + 7) / 8);
  }

  void resize(uint64_t _numItems, uint32_t _size_each) {
    numItems  = _numItems;
    size_each = _size_each;
    vec.resize((_numItems * _size_each + 7) / 8, false);
    std::fill(vec.begin(), vec.end(), 0);
    std::cerr << "resizing : " << vec[0] << "\n";
  }

  bool set_bit_and_return(uint64_t n, uint32_t a) {
    auto addr    = n * size_each + a;
    auto old_val = vec[addr / 8] & (1 << (addr % 8));
    vec[addr / 8] |= (1 << (addr % 8));
    return old_val;
  }

  void set_bit(uint64_t n, uint32_t a) {
    auto addr = n * size_each + a;
    vec[addr / 8] |= (1 << (addr % 8));
  }

  bool is_set(uint64_t n, uint32_t a) const {
    auto addr = n * size_each + a;
    return vec[addr / 8] & (1 << (addr % 8));
  }

  uint64_t size() const { return numItems; }

  uint32_t bit_count(uint64_t n) const {
    uint32_t set_bit_count = 0;
    for (uint32_t k = 0; k < size_each; ++k) {
      if (is_set(n, k)) {
        set_bit_count++;
      }
    }
    return set_bit_count;
  }
  uint32_t find_first(uint64_t n) const {
    for (uint32_t k = 0; k < size_each; ++k) {
      if (is_set(n, k)) {
        return k;
      }
    }
    return ~0;
  }
  uint32_t find_next(uint64_t n, uint32_t p) const {
    for (auto k = p + 1; k < size_each; ++k) {
      if (is_set(n, k)) {
        return k;
      }
    }
    return ~0;
  }

  void clear() { std::vector<uint8_t>().swap(vec); }
};

} // namespace galois
