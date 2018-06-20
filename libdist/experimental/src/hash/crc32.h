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
// crc32.h
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
typedef unsigned __int32 uint32_t;
#else
// GCC
#include <stdint.h>
#endif

/// compute CRC32 hash, based on Intel's Slicing-by-8 algorithm
/** Usage:
    CRC32 crc32;
    std::string myHash  = crc32("Hello World");     // std::string
    std::string myHash2 = crc32("How are you", 11); // arbitrary data, 11 bytes

    // or in a streaming fashion:

    CRC32 crc32;
    while (more data available)
      crc32.add(pointer to fresh data, number of new bytes);
    std::string myHash3 = crc32.getHash();

    Note:
    You can find code for the faster Slicing-by-16 algorithm on my website, too:
    http://create.stephan-brumme.com/crc32/
    Its unrolled version is about twice as fast but its look-up table doubled in
   size as well.
  */
class CRC32 //: public Hash
{
public:
  /// hash is 4 bytes long
  enum { HashBytes = 4 };

  /// same as reset()
  CRC32();

  /// compute CRC32 of a memory block
  std::string operator()(const void* data, size_t numBytes);
  /// compute CRC32 of a string, excluding final zero
  std::string operator()(const std::string& text);

  static uint32_t hash(const void* data, size_t numBytes);

  /// add arbitrary number of bytes
  void add(const void* data, size_t numBytes);

  /// return latest hash as 8 hex characters
  std::string getHash();
  /// return latest hash as bytes
  void getHash(unsigned char buffer[HashBytes]);

  /// restart
  void reset();

private:
  /// hash
  uint32_t m_hash;
};
