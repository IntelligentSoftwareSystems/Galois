/** Scheduler Microbenchmark -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * @section Description
 *
 * Test scalability of schedulers
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#include "Galois/Timer.h"
#include "Galois/Galois.h"

#include "Lonestar/BoilerPlate.h"

#ifdef GALOIS_USE_EXP
#include "Galois/WorkList/WorkListExperimental.h"
#endif

#include <iostream>

static const char* name = "Scheduler Micro Benchmark";
static const char* desc = "Measures stuff";
static const char* url = 0;

static llvm::cl::opt<int> sval(llvm::cl::Positional, llvm::cl::desc("<start value>"), llvm::cl::init(-1));
static llvm::cl::opt<int> ival(llvm::cl::Positional, llvm::cl::desc("<init num>"), llvm::cl::init(100));

struct process {
  void operator()(int item, galois::UserContext<int>& lwl) {
    for (int i = 0; i < item; ++i)
      lwl.push(item - 1);
  }
};

int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);

  std::vector<int> v((int)ival, (int)sval);

  std::cout << "Initial: " << (int)ival << " using " << (int)sval << "\n";

  // galois::StatTimer T0("T0");
  // T0.start();
  // using namespace galois::runtime::WorkList;
  // galois::for_each<ChunkedLIFO<64> >(v.begin(), v.end(), process());
  // T0.stop();

  // galois::StatTimer T1("T1");
  // T1.start();
  // using namespace galois::runtime::WorkList;
  // galois::for_each<Alt::ChunkedAdaptor<LIFO<>, 64, true > >(v.begin(), v.end(), process());
  // T1.stop();

  galois::StatTimer T2("T2");
  T2.start();
  using namespace galois::WorkList;
  galois::for_each(v.begin(), v.end(), process(), galois::wl<dChunkedLIFO<64>>());
  T2.stop();
}
