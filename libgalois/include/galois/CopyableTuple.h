/** Galois Contiguous Tuple -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
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
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#ifndef __GALOIS_COPYABLE_TUPLE__
#define __GALOIS_COPYABLE_TUPLE__

namespace galois {

/**
 * Struct that contains 3 elements. Used over std::tuple as std::tuple memory
 * layout isn't guaranteed.
 */
template<typename T1, typename T2, typename T3>
struct TupleOfThree {
  T1 first;
  T2 second;
  T3 third;

  // empty constructor
  TupleOfThree() { }

  // initialize 3 fields
  TupleOfThree(T1 one, T2 two, T3 three) {
    first = one;
    second = two;
    third = three;
  }
};

} // end galois namespace

#endif
