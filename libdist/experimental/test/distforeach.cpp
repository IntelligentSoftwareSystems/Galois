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
#include "galois/runtime/DistSupport.h"
#include <boost/iterator/counting_iterator.hpp>
#include <iostream>
#include <vector>

using namespace galois::runtime;

typedef std::vector<int>::iterator IterTy;

struct Counter : public galois::runtime::Lockable {
  int i;

  Counter() : i(0) {}
  Counter(RecvBuffer& buf) { deserialize(buf); }

  void add(int v) {
    std::cout << "Counter: in host " << NetworkInterface::ID << " and thread "
              << LL::getTID() << " processing number " << v << " old value "
              << i << std::endl; // endl for the flush
    i += v;
    return;
  }

  typedef int tt_has_serialize;
  void deserialize(RecvBuffer& buf) { gDeserialize(buf, i); }
  void serialize(SendBuffer& buf) const { gSerialize(buf, i); }
};

struct Adder {
  gptr<Counter> c;

  Adder(Counter* c = nullptr) : c(c) {}

  void operator()(int& data, galois::UserContext<int>&) {
    acquire(c, galois::ALL);
    c->add(data);
    return;
  }
  // Trivially_copyable
  typedef int tt_is_copyable;
};

struct Printer {
  void operator()(int& data, galois::UserContext<int>&) {
    std::cout << "Printer: in host " << NetworkInterface::ID << " and thread "
              << LL::getTID() << " processing number " << data << std::endl;
  }
};

void testSimple(int N) {
  auto begin = boost::counting_iterator<int>(0);
  auto end   = boost::counting_iterator<int>(N);

  galois::for_each(begin, end, Printer());
}

void testAdder(int N) {
  auto begin = boost::counting_iterator<int>(0);
  auto end   = boost::counting_iterator<int>(N);
  Counter c;
  Adder adder(&c);

  static_assert(galois::runtime::is_serializable<Counter>::value,
                "Counter not serializable");

  galois::for_each(begin, end, adder);

  acquire(adder.c, galois::ALL);
  auto val_is = adder.c->i, val_sb = std::accumulate(begin, end, 0);

  GALOIS_ASSERT(val_is == val_sb);
}

int main(int argc, char* argv[]) {
  galois::StatManager M;
  int threads = 2;
  if (argc > 1)
    threads = atoi(argv[1]);
  int N = 40;
  if (argc > 2)
    N = atoi(argv[2]);

  galois::setActiveThreads(threads);
  auto& net = galois::runtime::getSystemNetworkInterface();
  net.start();

  testSimple(N);
  testAdder(N);

  net.terminate();

  return 0;
}
