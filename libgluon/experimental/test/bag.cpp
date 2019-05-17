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

#include "galois/Galois.h"
#include "galois/graphs/Bag.h"
#include "Lonestar/BoilerPlate.h"

#include <boost/iterator/counting_iterator.hpp>

typedef galois::graphs::Bag<int>::pointer IntPtrs;

struct InsertBody {
  IntPtrs pBodies;

  template <typename Context>
  void operator()(int i, const Context& ctx) {
    galois::runtime::LL::gPrint("host: ", galois::runtime::NetworkInterface::ID,
                                " pushing: ", i, "\n");
    pBodies->push(i);
  }

  // Trivially_copyable
  typedef int tt_is_copyable;
};

struct PrintInt {
  template <typename Context>
  void operator()(int i, Context& ctx) {
    galois::runtime::LL::gPrint("host: ", galois::runtime::NetworkInterface::ID,
                                " received: ", i, "\n");
  }
};

int main(int argc, char** argv) {
  LonestarStart(argc, argv, nullptr, nullptr, nullptr);
  galois::runtime::getSystemNetworkInterface().start();

  IntPtrs pBodies = galois::graphs::Bag<int>::allocate();
  galois::for_each(boost::counting_iterator<int>(0),
                   boost::counting_iterator<int>(10), InsertBody{pBodies});
  galois::for_each(pBodies, PrintInt());

  galois::runtime::getSystemNetworkInterface().terminate();

  return 0;
}
