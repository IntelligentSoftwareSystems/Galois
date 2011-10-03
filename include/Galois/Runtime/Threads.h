/** Simple thread related classes -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_THREADS_H
#define GALOIS_RUNTIME_THREADS_H

#include <vector>

#include "Galois/Runtime/Config.h"

namespace GaloisRuntime {

struct runCMD {
  config::function<void (void)> work;
  bool isParallel;
  bool barrierAfter;
};

class ThreadPool {
protected:
  static __thread unsigned int LocalThreadID;
  unsigned int activeThreads;

public:

  //!execute work on all threads
  //!preWork and postWork are executed only on the master thread
  virtual void run(runCMD* begin, runCMD* end) = 0;
  
  //!change the number of threads to num
  //!returns the number that the runtime chooses (may not be num)
  virtual unsigned int setActiveThreads(unsigned int num) = 0;

  //!How many threads will be used
  unsigned int getActiveThreads() const { return activeThreads; }

  //!My thread id (dense, user thread is 0, galois threads 1..num)
  static unsigned int getMyID() { return LocalThreadID; }

};

//Returns or creates the appropriate thread pool for the system
ThreadPool& getSystemThreadPool();

class ThreadPolicy {
protected:
  const char* name;

  //number of hw supported threads
  int numThreads;
  
  //number of "real" processors
  int numCores;

  //number of packages
  int numPackages;

  //number of threads per core
  int htRatio;

  //number of threads in each level
  // it is assumed that (thread id % numCores) / levelSize == bin for that level
  // this works because threads are densely numbered in this way
  //commonly this is structured as:
  // level[0] = non-SMT Threads/core (Each core is it's own thing)
  // level[1] = non-SMT Threads/L3 (thread to L3 mapping)
  // level[2] = non-SMT Threads/NUMA Node (optional)
  std::vector<int> levelSize;

public:
  const char* getName() const { return name; }

  //Return the bin for thread thr at level level
  int indexLevelMap(int level, int thr) const {
    return (thr % numCores) / levelSize[level];
  }

  int getNumLevels() const { return levelSize.size(); }

  int getNumThreads() const { return numThreads; }

  int getNumCores() const { return numCores; }

  int getLevelBins(int level) const { return numCores / levelSize[level]; }

  int isFirstInLevel(int level, int thr) const {
    return thr % levelSize[level] == 0 &&
      thr / numCores == 0;
  }

  virtual void bindThreadToProcessor() = 0;
};

ThreadPolicy& getSystemThreadPolicy();

}

#endif
