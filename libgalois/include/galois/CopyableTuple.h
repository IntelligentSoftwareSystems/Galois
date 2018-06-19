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
 * @file CopyableTuple.h
 *
 * Contains copyable tuple classes whose elements are contiguous in memory
 */
#ifndef __GALOIS_COPYABLE_TUPLE__
#define __GALOIS_COPYABLE_TUPLE__

namespace galois {

/**
 * Struct that contains 3 elements. Used over std::tuple as std::tuple memory
 * layout isn't guaranteed.
 *
 * @tparam T1 type of first element
 * @tparam T2 type of second element
 * @tparam T3 type of third element
 */
template <typename T1, typename T2, typename T3>
struct TupleOfThree {
  //! first element
  T1 first;
  //! second element
  T2 second;
  //! third element
  T3 third;

  //! empty constructor
  TupleOfThree() {}

  //! Constructor that initializes 3 fields
  TupleOfThree(T1 one, T2 two, T3 three) {
    first  = one;
    second = two;
    third  = three;
  }
};

} // namespace galois

#endif
