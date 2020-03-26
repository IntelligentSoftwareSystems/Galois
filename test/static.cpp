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

// std_tr1__type_traits__is_pod.cpp

#include "galois/substrate/PtrLock.h"
#include "galois/substrate/SimpleLock.h"
#include "galois/substrate/StaticInstance.h"

#include <type_traits>
#include <iostream>

using namespace galois::substrate;

int main() {
  std::cout << "is_pod PtrLock<int> == " << std::boolalpha
            << std::is_pod<PtrLock<int>>::value << std::endl;

  std::cout << "is_pod SimpleLock == " << std::boolalpha
            << std::is_pod<SimpleLock>::value << std::endl;
  std::cout << "is_pod DummyLock == " << std::boolalpha
            << std::is_pod<DummyLock>::value << std::endl;

  std::cout << "is_pod StaticInstance<int> == " << std::boolalpha
            << std::is_pod<StaticInstance<int>>::value << std::endl;
  std::cout << "is_pod StaticInstance<std::iostream> == " << std::boolalpha
            << std::is_pod<StaticInstance<std::iostream>>::value << std::endl;

  std::cout << "is_pod volatile int == " << std::boolalpha
            << std::is_pod<volatile int>::value << std::endl;
  std::cout << "is_pod int == " << std::boolalpha << std::is_pod<int>::value
            << std::endl;

  // std::cout << "is_pod<throws> == " << std::boolalpha
  // 	    << std::is_pod<throws>::value << std::endl;

  return (0);
}
