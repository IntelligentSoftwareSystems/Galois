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

/**
 * @file ArrayWrapper.h
 *
 * Defines the CopyableArray subclass used to make arrays trivially copyable if
 * possible.
 */

#ifndef _ARRAY_WRAPPER_H_
#define _ARRAY_WRAPPER_H_

#include <array>
#include "galois/runtime/ExtraTraits.h"

namespace galois {
/**
 * A subclass of std::array that is marked trivially copyable if the type is
 * also memory copyable. Useful when you need a trivially copyable type for
 * serialization.
 *
 * @tparam T type of the items to be stored in the array
 * @tparam N total number of items in the array
 */
template <class T, size_t N>
class CopyableArray : public std::array<T, N> {
public:
  //! Only typedef tt_is_copyable if T is trivially copyable.
  //! Allows the use of memcopy in serialize/deserialize.
  using tt_is_copyable =
      typename std::enable_if<galois::runtime::is_memory_copyable<T>::value,
                              int>::type;
};
} // namespace galois
#endif
