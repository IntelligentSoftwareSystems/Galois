/** micro-benchmark for numa accesses -*- C++ -*-
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
 * Micro benchmark for numa accesses.
 *
 * @author Yi-Shan Lu <yishanlu@cs.utexas.edu>
 */

#include "Galois/Statistic.h"
#include "Galois/Galois.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/LargeArray.h"
#include "Galois/Substrate/Barrier.h"

#include <iostream>
#include <chrono>
#include <random>

const char* name = "Micro Benchmark for NUMA accesses";
const char* desc = "Test for the behavior of NUMA accesses";
const char* url = 0;

namespace cll = llvm::cl;
static cll::opt<unsigned int> numArrayEntry("numArrayEntry", cll::desc("number of entries for the NUMA array"), cll::init(20000000000U));
static cll::opt<unsigned int> timesAccessed("timesAccessed", cll::desc("number of accesses to the NUMA array"), cll::init(10000000U));
static cll::opt<unsigned int> chunkSize("chunkSize", cll::desc("number of consecutive entries accessed per visit"), cll::init(1U));

typedef int Element;
typedef Galois::LargeArray<Element> Array;

struct NumaAccess {
  Array& array;
  NumaAccess(Array& _a): array(_a) {}

  void operator()(int tid, int numThreads) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + (unsigned int)tid;
    std::minstd_rand0 generator(seed);
    std::uniform_int_distribution<int> distribution(0, numArrayEntry);

    int accesses = timesAccessed / numThreads;
    if(tid < timesAccessed % numThreads) {
      accesses++;
    }

    int numChunk = accesses / (int)chunkSize;
    int lastChunk = accesses % (int) chunkSize;
    if(lastChunk == 0) {
      lastChunk = chunkSize;
      numChunk--;
    }

    // eliminate uneven thread launching overhead
    Galois::Substrate::Barrier& barrier = Galois::Runtime::getBarrier(numThreads);
    barrier.wait();

    Galois::StatTimer timer("AccessTime");
    timer.start();

    int start = distribution(generator);
    for(int j = 0; j < lastChunk; j++) {
      array[(start+j)%numArrayEntry] = (Element)tid;
    }

    for(int i = 0; i < numChunk; i++) {
      start = distribution(generator);
      for(int j = 0; j < (int)chunkSize; j++) {
        array[(start+j)%numArrayEntry] = (Element)tid;
      }
    }

    timer.stop();
  }
};

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  Galois::StatTimer totalTime("TotalTime");
  Galois::StatTimer T;
  
  totalTime.start();
  Array array;
  array.allocateInterleaved(numArrayEntry);

  T.start();
  Galois::on_each(NumaAccess(array), Galois::loopname("NumaAccess"));
  T.stop();

  totalTime.stop();
  return 0;
}

