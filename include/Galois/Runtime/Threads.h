// Simple Thread Related Classes -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

#ifndef _GALOISRUNTIME_THREADS_H
#define _GALOISRUNTIME_THREADS_H

#include "Galois/Executable.h"

#include <boost/intrusive/list.hpp>
#include <vector>

namespace GaloisRuntime {

//declared out of line to correctly initialize data in Threads.cpp
struct initMainThread {
  initMainThread();
};

class ThreadPool {
  friend struct initMainThread;
  static __thread unsigned int LocalThreadID;
  static int nextThreadID;
protected:
  unsigned int activeThreads;

protected:
  static void NotifyAware(bool starting);

public:
  //!execute work on all threads
  //!The work object is not duplicated in any way 
  virtual void run(Galois::Executable* work) = 0;
  
  //!change the number of threads to num
  //!returns the number that the runtime chooses (may not be num)
  virtual unsigned int setActiveThreads(unsigned int num) = 0;

  //!How many threads will be used
  unsigned int getActiveThreads() const { return activeThreads; }

  static unsigned int getMyID() __attribute__((pure));

};

//Returns or creates the appropriate thread pool for the system
ThreadPool& getSystemThreadPool();

class ThreadPolicy {
protected:
  //num levels
  int numLevels;
  
  //number of hw supported threads
  int numThreads;
  
  //number of "real" processors
  int numCores;

  //example levels:
  //thread(0), Cpu(1), numa(2), machine(3)

  //Total number of threads in each level
  std::vector<int> levelSize;

  //[numLevels][numThreads] -> item index for thread at level
  std::vector<int> levelMap;

public:
  int indexLevelMap(int level, int thr) const {
    return levelMap[level * numThreads + thr];
  }

  int getNumLevels() const { return numLevels; }

  int getNumThreads() const { return numThreads; }

  int getNumCores() const { return numCores; }

  int getLevelSize(int S) const { return levelSize[S]; }

  int isFirstInLevel(int level, int thr) const {
    int thrLevel = indexLevelMap(level, thr);
    for (int i = 0; i < getNumThreads(); ++i)
      if (indexLevelMap(level, i) == thrLevel)
	return i == thr;
    //Should be dead:
    return false;
  }

  virtual void bindThreadToProcessor(int id) = 0;
};

ThreadPolicy& getSystemThreadPolicy();
void setSystemThreadPolicy(const char* name);

namespace HIDDEN {
//Tag for invasive list
class ThreadAwareTag;

//Hook type for invasive list
typedef boost::intrusive::list_base_hook<boost::intrusive::tag<ThreadAwareTag> > ThreadAwareHook;
}

//This notifies when the number of threads change
class ThreadAware : public HIDDEN::ThreadAwareHook {
  friend class ThreadPool;
  static void NotifyOfChange(bool starting);

public:
  ThreadAware();
  virtual ~ThreadAware();

  //This is called to notify the start and end of a parallel region
  //starting = true -> parallel code is initializing
  //starting = false -> parallel code is ending
  virtual void ThreadChange(bool starting) = 0;

};

}

#endif
