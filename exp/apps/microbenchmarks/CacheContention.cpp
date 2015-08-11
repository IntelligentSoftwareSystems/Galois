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
 * Micro benchmark for cache contention.
 *
 * @author Yi-Shan Lu <yishanlu@cs.utexas.edu>
 */

#include "Galois/Statistic.h"
#include "Galois/Galois.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/Substrate/Barrier.h"

#include <iostream>

const char* name = "Micro Benchmark for Cache Contention";
const char* desc = "Test for the behavior of cache contention";
const char* url = 0;

enum Algo {
  writeGlobal,
  nonAtomicRMW,
  fetchAndAdd,
  cas, 
  casWithPause
};

namespace cll = llvm::cl;
static cll::opt<unsigned int> timesAdd("timesAdd", cll::desc("number of times to increment a global counter"), cll::init(100000000U));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
  cll::values(
    clEnumValN(Algo::writeGlobal, "writeGlobal", "write to a global counter"),
    clEnumValN(Algo::nonAtomicRMW, "nonAtomicRMW", "non-atomic read-modify-write"), 
    clEnumValN(Algo::fetchAndAdd, "fetchAndAdd", "fetch and add"), 
    clEnumValN(Algo::cas, "cas", "compare and swap (default)"),
    clEnumValN(Algo::casWithPause, "casWithPause", "cas with pause when failed"),
    clEnumValEnd), cll::init(Algo::cas));

static unsigned int globalAtomicCounter = 0;
static volatile unsigned int globalCounter = 0;

struct Contention {
  void operator()(int tid, int numThreads) {
    unsigned int upper = timesAdd / numThreads;
    if(tid < timesAdd % numThreads) {
      upper += 1;
    }

    // to eliminate uneven thread launching overhead 
    Galois::Substrate::Barrier& barrier = Galois::Substrate::getSystemBarrier(numThreads);
    barrier.wait();

    switch (algo) {
    case Algo::writeGlobal:
      for(unsigned int i = 0; i < upper; i++) {
        globalCounter = i;
      }
      break;
    case Algo::nonAtomicRMW:
      for(unsigned int i = 0; i < upper; i++) {
        globalCounter += 1;
      }
      break;
    case Algo::fetchAndAdd:
      for(unsigned int i = 0; i < upper; i++) {
        __sync_fetch_and_add(&globalAtomicCounter, 1);
      }
      break;
    case Algo::cas:
      for(unsigned int i = 0; i < upper; i++) {
        while(true) {
          unsigned int oldV = globalAtomicCounter;
          unsigned int newV = oldV + 1;
          if(__sync_bool_compare_and_swap(&globalAtomicCounter, oldV, newV)) {
            break;
          }
        }
      }
      break;
    case Algo::casWithPause:
    default:
      for(unsigned int i = 0; i < upper; i++) {
        while(true) {
          unsigned int oldV = globalAtomicCounter;
          unsigned int newV = oldV + 1;
          if(__sync_bool_compare_and_swap(&globalAtomicCounter, oldV, newV)) {
            break;
          } else {
            asm volatile("pause\n": : :"memory");
          }
        }
      }
      break;
    } // end switch (algo)
  }
};

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  Galois::StatTimer T;
  T.start();
  Galois::on_each(Contention(), Galois::loopname("Contention"));
  T.stop();

  if(algo == Algo::nonAtomicRMW || algo == Algo::writeGlobal) {
    std::cout << "globalCounter = " << globalCounter << std::endl;
  } else {
    std::cout << "globalAtomicCounter = " << globalAtomicCounter << std::endl;
  }
  return 0;
}

