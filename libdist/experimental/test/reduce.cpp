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
#include "galois/graphs/Graph3.h"
#include "galois/Reduction.h"
#include "galois/runtime/PerHostStorage.h"
#include "Lonestar/BoilerPlate.h"

#include <boost/iterator/counting_iterator.hpp>

#include <iostream>

using namespace galois::graphs;
using namespace galois::runtime;

typedef galois::DGReducible<unsigned, std::plus<unsigned>> RD1;

struct op1 {
  gptr<RD1> count;

  op1(gptr<RD1> c) : count(c) {}
  op1() {}

  void operator()(const int& nodeval, const galois::UserContext<int>&) {
    count->get() += 1;
  }

  typedef int tt_has_serialize;
  void serialize(SerializeBuffer& s) const { gSerialize(s, count); }
  void deserialize(DeSerializeBuffer& s) { gDeserialize(s, count); }
};

void check1() {
  gptr<RD1> Cr(new RD1());

  std::cout << "Loop\n";
  galois::for_each<>(boost::counting_iterator<int>(0),
                     boost::counting_iterator<int>(20), op1(Cr));
  std::cout << "\n";

  for (int i = 0; i < 4; ++i) {
    std::cout << Cr->get() << "\n";
    std::cout << "Reduce\n";
    unsigned& x = Cr->doReduce();
    std::cout << Cr->get() << " " << x << "\n";
    std::cout << "Broadcast\n";
    Cr->doBroadcast(x);
    std::cout << Cr->get() << " " << x << "\n";
  }
}

typedef galois::DGReducibleVector<int, std::plus<int>> RD2;

struct Show {
  gptr<RD2> count;
  Show() {}
  Show(gptr<RD2> c) : count(c) {}

  void operator()(unsigned tid, unsigned) {
    for (int i = 0; i < 20; ++i) {
      std::cout << networkHostID << " " << tid << " c[" << i
                << "]: " << count->get(i) << "\n";
    }
  }

  typedef int tt_has_serialize;
  void serialize(SerializeBuffer& s) const { gSerialize(s, count); }
  void deserialize(DeSerializeBuffer& s) { gDeserialize(s, count); }
};

struct op2 {
  gptr<RD2> count;
  op2() {}
  op2(gptr<RD2> c) : count(c) {}

  void operator()(int n, galois::UserContext<int>&) { count->update(n, n); }

  typedef int tt_has_serialize;
  void serialize(SerializeBuffer& s) const { gSerialize(s, count); }
  void deserialize(DeSerializeBuffer& s) { gDeserialize(s, count); }
};

void check2() {
  gptr<RD2> p(new RD2());
  p->allocate(20);

  galois::for_each(boost::counting_iterator<int>(0),
                   boost::counting_iterator<int>(20), op2(p));
  galois::for_each(boost::counting_iterator<int>(0),
                   boost::counting_iterator<int>(20), op2(p));
  for (int i = 0; i < 20; ++i) {
    std::cout << "local[" << i << "]: " << p->get(i) << "\n";
  }
  p->doReduce();
  for (int i = 0; i < 20; ++i) {
    std::cout << "reduced[" << i << "]: " << p->get(i) << "\n";
  }

  // galois::on_each(Show(p));
  p->doReset();
  distWait();

  galois::for_each(boost::counting_iterator<int>(0),
                   boost::counting_iterator<int>(20), op2(p));
  galois::for_each(boost::counting_iterator<int>(0),
                   boost::counting_iterator<int>(20), op2(p));
  for (int i = 0; i < 20; ++i) {
    std::cout << "local[" << i << "]: " << p->get(i) << "\n";
  }
  p->doAllReduce();
  for (int i = 0; i < 20; ++i) {
    std::cout << "reduced[" << i << "]: " << p->get(i) << "\n";
  }

  distWait();
  //  galois::runtime::deallocatePerHost(p);
  delete &*p;
}

int main(int argc, char** argv) {
  LonestarStart(argc, argv, nullptr, nullptr, nullptr);
  networkStart();

  std::cout << "====Check1====\n";
  check1();
  std::cout << "====Check2====\n";
  check2();

  networkTerminate();
  return 0;
}
