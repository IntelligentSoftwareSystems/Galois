// -*- C++ -*-
/*! \file 
 *  \brief simple thread related classes
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
  static __thread int LocalThreadID;
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
  int indexLevelMap(int level, int thr) {
    return levelMap[level * numThreads + thr];
  }

  int getNumLevels() const { return numLevels; }

  int getNumThreads() const { return numThreads; }

  int getNumCores() const { return numCores; }

  int getLevelSize(int S) const { return levelSize[S]; }

  virtual void bindThreadToProcessor(int id) = 0;
};

ThreadPolicy& getSystemThreadPolicy();

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
