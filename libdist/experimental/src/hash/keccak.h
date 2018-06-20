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

// //////////////////////////////////////////////////////////
// keccak.h
// Copyright (c) 2014,2015 Stephan Brumme. All rights reserved.
// see http://create.stephan-brumme.com/disclaimer.html
//

#pragma once

//#include "hash.h"
#include <string>

// define fixed size integer types
#ifdef _MSC_VER
// Windows
typedef unsigned __int8 uint8_t;
typedef unsigned __int64 uint64_t;
#else
// GCC
#include <stdint.h>
#endif

/// compute Keccak hash (designated SHA3)
/** Usage:
    Keccak keccak;
    std::string myHash  = keccak("Hello World");     // std::string
    std::string myHash2 = keccak("How are you", 11); // arbitrary data, 11 bytes

    // or in a streaming fashion:

    Keccak keccak;
    while (more data available)
      keccak.add(pointer to fresh data, number of new bytes);
    std::string myHash3 = keccak.getHash();
  */
class Keccak //: public Hash
{
public:
  /// algorithm variants
  enum Bits {
    Keccak224 = 224,
    Keccak256 = 256,
    Keccak384 = 384,
    Keccak512 = 512
  };

  /// same as reset()
  explicit Keccak(Bits bits = Keccak256);

  /// compute hash of a memory block
  std::string operator()(const void* data, size_t numBytes);
  /// compute hash of a string, excluding final zero
  std::string operator()(const std::string& text);

  /// add arbitrary number of bytes
  void add(const void* data, size_t numBytes);

  /// return latest hash as hex characters
  std::string getHash();

  /// restart
  void reset();

private:
  /// process a full block
  void processBlock(const void* data);
  /// process everything left in the internal buffer
  void processBuffer();

  /// 1600 bits, stored as 25x64 bit, BlockSize is no more than 1152 bits
  /// (Keccak224)
  enum { StateSize = 1600 / (8 * 8), MaxBlockSize = 200 - 2 * (224 / 8) };

  /// hash
  uint64_t m_hash[StateSize];
  /// size of processed data in bytes
  uint64_t m_numBytes;
  /// block size (less or equal to MaxBlockSize)
  size_t m_blockSize;
  /// valid bytes in m_buffer
  size_t m_bufferSize;
  /// bytes not processed yet
  uint8_t m_buffer[MaxBlockSize];
  /// variant
  Bits m_bits;
};
