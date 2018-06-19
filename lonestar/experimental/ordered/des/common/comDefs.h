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

#ifndef DES_COM_DEFS_H
#define DES_COM_DEFS_H

#include <limits>
#include <string>
#include <algorithm>
#include <vector>

namespace des {

/**
 * type for time in simulation world
 */
typedef long long SimTime;

// const SimTime INFINITY_SIM_TIME = std::numeric_limits<SimTime>::max ();
// The above definition is bad because INFINITY_SIM_TIME + small_value will
// cause an overflow and the result is not INFINITY_SIM_TIME any more

/** The Constant INFINITY_SIM_TIME is used by NULL_EVENT messages to signal the
 * end of simulation. */
const SimTime INFINITY_SIM_TIME = (1 << 30);

const SimTime MIN_DELAY = 1l;

/**
 * Helper function to convert a string to lower case
 */
std::string toLowerCase(std::string str);

/**
 * freeing pointers in a vector
 * before the vector itself is destroyed
 */
template <typename T>
void destroyVec(std::vector<T*>& vec) {
  for (typename std::vector<T*>::iterator i = vec.begin(), ei = vec.end();
       i != ei; ++i) {
    delete *i;
    *i = NULL;
  }
  vec.clear();
}

enum NullEventOpt { NEEDS_NULL_EVENTS, NO_NULL_EVENTS };

} // namespace des

#endif
