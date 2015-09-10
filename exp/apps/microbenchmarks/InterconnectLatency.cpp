/** micro-benchmark for cache contention -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights reserved.
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
 * Micro benchmark for interconnect latency.
 *
 * @author Yi-Shan Lu <yishanlu@cs.utexas.edu>
 */

#include "Galois/Statistic.h"
#include "Galois/Galois.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/Substrate/Barrier.h"

#include <iostream>

const char* name = "Micro Benchmark for Interconnect Latency";
const char* desc = "Test for the latency for interconnects";
const char* url = 0;

namespace cll = llvm::cl;
static cll::opt<unsigned int> timesAdd("timesAdd", cll::desc("number of times to increment a global counter"), cll::init(10000000U));
static cll::opt<unsigned int> firstCore("firstCore", cll::desc("the index of the core for the first thread"), cll::init(0U));
static cll::opt<unsigned int> secondCore("secondCore", cll::desc("the index of the core for the second thread"), cll::init(1U));

static volatile unsigned int globalAtomicCounter = 0;

struct PingPong {
  Galois::Substrate::Barrier& barrier;

  PingPong(Galois::Substrate::Barrier& b): barrier(b) {}

  void operator()(int tid, int numThreads) {
    unsigned int inc = (firstCore == secondCore) ? 1 : 2;
    unsigned int pos = (tid == firstCore) ? 0 : 1;
    unsigned int upper = 
        (tid == firstCore) ? 
          ((firstCore == secondCore) ? timesAdd : (timesAdd/2+timesAdd%2)) : 
          ((tid == secondCore) ? (timesAdd/2) : 0);

    // to eliminate uneven thread launching overhead 
    barrier.wait();

    for(unsigned int i = 0; i < upper; i++) {
      while(globalAtomicCounter != pos) {} // wait until the other thread performs one atomic addition
      __sync_fetch_and_add(&globalAtomicCounter, 1);
      pos += inc;
    }
  }
};

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  std::cout << "measuring delay in between core " << firstCore << " and core " << secondCore << std::endl;

  Galois::StatTimer T;
  T.start();
  Galois::on_each(PingPong(Galois::Runtime::getBarrier(Galois::Runtime::activeThreads)), Galois::loopname("PingPong"));
  T.stop();

  std::cout << "globalAtomicCounter = " << globalAtomicCounter << std::endl;
  return 0;
}

