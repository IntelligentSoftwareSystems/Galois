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

#include "Galois/Timer.h"
#include "Galois/Galois.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/LargeArray.h"
#include "Galois/Substrate/Barrier.h"
#include "Galois/Substrate/ThreadPool.h"
#include "Galois/Substrate/PerThreadStorage.h"

#include <iostream>
#include <chrono>
#include <random>
#include <cstring>

#include <numa.h>

const char* name = "Micro Benchmark for NUMA accesses";
const char* desc = "Test for the behavior of NUMA accesses";
const char* url = 0;

enum Algo {
  globalInterleaved,
  globalLarge,
  local,
  starInterleavedGraphRead,
  starLargeGraphRead
};

namespace cll = llvm::cl;
static cll::opt<unsigned int> numArrayEntry("numArrayEntry", cll::desc("number of entries for the NUMA array"), cll::init(20000000000U));
static cll::opt<unsigned int> timesAccessed("timesAccessed", cll::desc("number of accesses to the NUMA array"), cll::init(10000000U));
static cll::opt<unsigned int> chunkSize("chunkSize", cll::desc("number of consecutive entries accessed per visit"), cll::init(1U));
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"),
  cll::values(
    clEnumValN(Algo::globalInterleaved, "globalInterleaved", "global numa accesses on interleavedly allocated array (default)"),
    clEnumValN(Algo::globalLarge, "globalLarge", "global numa accesses on locally allocated array"),
    clEnumValN(Algo::local, "local", "local accesses"),
    clEnumValN(Algo::starInterleavedGraphRead, "starInterleavedGraphRead", "repeatedly write local=(global+tid) to random interleavedly allocated array elements"), 
    clEnumValN(Algo::starLargeGraphRead, "starLargeGraphRead", "repeatedly write local=(global+tid) to random locally allocated array elements"),
    clEnumValEnd), cll::init(Algo::globalInterleaved));

typedef unsigned int Element;
typedef Galois::LargeArray<Element> Array;

static volatile Element globalCounter = 0;

struct StarGraphRead {
  Array& array;
  Galois::Substrate::Barrier& barrier;
  StarGraphRead(Array& _a, Galois::Substrate::Barrier& b): array(_a), barrier(b) {}

  void operator()(int tid, int numThreads) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + (unsigned)tid;
    std::minstd_rand0 generator(seed);
    std::uniform_int_distribution<unsigned> distribution(0, numArrayEntry);

    unsigned accesses = timesAccessed / (unsigned)numThreads;
    if(tid < timesAccessed % (unsigned)numThreads) {
      accesses++;
    }

    // eliminate uneven thread launching overhead
    barrier.wait();

    Galois::StatTimer timer("AccessTime");
    timer.start();

    for(unsigned i = 0; i < accesses; i++) {
      Element* pos = &array[distribution(generator)%numArrayEntry];
      Element local = globalCounter + (Element)tid; // intended global read and dependency
      while(true) {
        Element old = *pos; 
        if(__sync_bool_compare_and_swap(pos, old, local)) {
          break;
        }
      }
    }

    timer.stop();
  }
};

struct GlobalAccess {
  Array& array;
  Galois::Substrate::Barrier& barrier;
  GlobalAccess(Array& _a, Galois::Substrate::Barrier& b): array(_a), barrier(b) {}

  void operator()(int tid, int numThreads) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + (unsigned)tid;
    std::minstd_rand0 generator(seed);
    std::uniform_int_distribution<unsigned> distribution(0, numArrayEntry);

    unsigned accesses = timesAccessed / (unsigned)numThreads;
    if(tid < timesAccessed % (unsigned)numThreads) {
      accesses++;
    }

    unsigned numChunk = accesses / chunkSize;
    unsigned lastChunk = accesses % chunkSize;
    if(lastChunk == 0) {
      lastChunk = chunkSize;
      numChunk--;
    }

    // eliminate uneven thread launching overhead
    barrier.wait();

    Galois::StatTimer timer("AccessTime");
    timer.start();

    unsigned start = distribution(generator);
    for(unsigned j = 0; j < lastChunk; j++) {
      array[(start+j)%numArrayEntry] = (Element)tid;
    }

    for(unsigned i = 0; i < numChunk; i++) {
      start = distribution(generator);
      for(unsigned j = 0; j < chunkSize; j++) {
        array[(start+j)%numArrayEntry] = (Element)tid;
      }
    }

    timer.stop();
  }
};

struct LocalArray {
  Element* array;
  unsigned numEntry;
  LocalArray(unsigned nE): numEntry(nE) {}

  void alloc() {
    array = (Element*)numa_alloc_local(sizeof(Element)*numEntry);
    memset(array, 0, sizeof(Element)*numEntry); // touch to ensure physical allocation
  }

  void dealloc() {
    numa_free((void*)array, sizeof(Element)*numEntry);
  }
};

typedef Galois::Substrate::PerPackageStorage<LocalArray> PerPackageArray;

struct LocalAccess {
  PerPackageArray& pArray;
  Galois::Substrate::Barrier& barrier;
  LocalAccess(PerPackageArray& _a, Galois::Substrate::Barrier& b): pArray(_a), barrier(b) {}

  void operator()(int tid, int numThreads) {
    unsigned numEntry = pArray.getLocal()->numEntry;
    Element* array = pArray.getLocal()->array;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + (unsigned)tid;
    std::minstd_rand0 generator(seed);
    std::uniform_int_distribution<unsigned> distribution(0, numEntry);

    unsigned accesses = timesAccessed / (unsigned)numThreads;
    if(tid < timesAccessed % (unsigned)numThreads) {
      accesses++;
    }

    unsigned numChunk = accesses / chunkSize;
    unsigned lastChunk = accesses % chunkSize;
    if(lastChunk == 0) {
      lastChunk = chunkSize;
      numChunk--;
    }

    // eliminate uneven thread launching overhead
    barrier.wait();

    Galois::StatTimer timer("AccessTime");
    timer.start();

    unsigned start = distribution(generator);
    for(unsigned j = 0; j < lastChunk; j++) {
      array[(start+j)%numEntry] = (Element)tid;
    }

    for(unsigned i = 0; i < numChunk; i++) {
      start = distribution(generator);
      for(unsigned j = 0; j < chunkSize; j++) {
        array[(start+j)%numEntry] = (Element)tid;
      }
    }

    timer.stop();
  }
};

struct LocalMalloc {
  PerPackageArray& pArray;
  LocalMalloc(PerPackageArray& _a): pArray(_a) {}

  void operator()(int tid, int numThreads) {
    if(Galois::Substrate::ThreadPool::isLeader()) {
      pArray.getLocal()->alloc();
    }
  }
};

struct LocalFree {
  PerPackageArray& pArray;
  LocalFree(PerPackageArray& _a): pArray(_a) {}

  void operator()(int tid, int numThreads) {
    if(Galois::Substrate::ThreadPool::isLeader()) {
      pArray.getLocal()->dealloc();
    }
  }
};

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  Galois::StatTimer totalTime("TotalTime");
  Galois::StatTimer T;
  
  totalTime.start();
  Array array;
  unsigned numPackageUsed = Galois::Substrate::getThreadPool().getCumulativeMaxPackage(Galois::Runtime::activeThreads)+1;
  PerPackageArray perPackageArray(numArrayEntry/numPackageUsed);

  if(algo == Algo::local) {
    Galois::on_each(LocalMalloc(perPackageArray), Galois::loopname("LocalMalloc"));
  } else if(algo == Algo::globalLarge || algo == starLargeGraphRead) {
    array.allocateLocal(numArrayEntry);
  } else {
    array.allocateInterleaved(numArrayEntry);
  }

  T.start();
  auto& barrier = Galois::Runtime::getBarrier(Galois::Runtime::activeThreads);
  if(algo == Algo::starInterleavedGraphRead || algo == Algo::starLargeGraphRead) {
    Galois::on_each(StarGraphRead(array, barrier), Galois::loopname("StarGraphRead"));
  } else if(algo == Algo::local) {
    Galois::on_each(LocalAccess(perPackageArray, barrier), Galois::loopname("LocalAccess"));
  } else {
    Galois::on_each(GlobalAccess(array, barrier), Galois::loopname("GlobalAccess"));
  }
  T.stop();

  if(algo == Algo::local) {
    Galois::on_each(LocalFree(perPackageArray), Galois::loopname("LocalFree"));
  }

  totalTime.stop();
  return 0;
}

