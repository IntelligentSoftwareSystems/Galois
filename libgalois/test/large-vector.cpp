/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2024, The University of Texas at Austin. All rights reserved.
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

#include <iostream>

#include "galois/Galois.h"
#include "galois/LargeVector.h"

int main() {
  galois::SharedMemSys Galois_runtime;

  {
    galois::LargeVector<uint64_t> the_vector;

    // should use 4 hugepages
    std::vector<uint64_t*> refs;
    for (size_t i = 0; i < (1 << 21); ++i) {
      refs.emplace_back(&the_vector.emplace_back(i));
    }

    for (size_t i = 0; i < (1 << 21); ++i) {
      GALOIS_ASSERT(*refs[i] == i);
    }
  }

  {
    static uint64_t num_constructed = 0, num_destructed = 0;
    class Object {
      uint8_t dummy;

    public:
      Object() { ++num_constructed; }
      ~Object() { ++num_destructed; }
    };
    static_assert(sizeof(Object) > 0);

    const size_t max_cap = (1 << 22);
    galois::LargeVector<Object> the_vector(max_cap);
    // constructor should not actually fill the vector
    GALOIS_ASSERT(num_constructed == 0);

    // entire vector should be mapped, even if it is empty
    const Object* addr = &the_vector[max_cap];
    GALOIS_ASSERT((addr - &the_vector[0]) == max_cap);

    the_vector.resize(max_cap);

    GALOIS_ASSERT(num_constructed == max_cap);
    GALOIS_ASSERT(addr == &the_vector[max_cap]);

    // resize should call the destructor, but vector should stay mapped
    GALOIS_ASSERT(num_destructed == 0);
    the_vector.resize(0);
    GALOIS_ASSERT(num_destructed == max_cap);
    GALOIS_ASSERT(addr == &the_vector[max_cap]);
  }

  return 0;
}
