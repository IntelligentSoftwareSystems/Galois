// -*- C++ -*-
/*! \file 
 *  \brief simple thread related classes
 */

#ifndef _GALOISRUNTIME_THREADS_H
#define _GALOISRUNTIME_THREADS_H

#include "Galois/Executable.h"

#include <boost/intrusive/list.hpp>

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
  unsigned int maxThreads;

protected:
  static void NotifyAware(bool starting);

public:
  //!execute work on all threads
  //!The work object is not duplicated in any way 
  virtual void run(Galois::Executable* work) = 0;
  
  //!change the number of preallocated threads to num
  //!returns the number that the runtime chooses (may not be num)
  virtual unsigned int setMaxThreads(unsigned int num) = 0;
  
  //!How many threads are kept around
  unsigned int getMaxThreads() const { return maxThreads; }

  //!How many threads will be used
  unsigned int getActiveThreads() const { return activeThreads; }

  static unsigned int getMyID();

};

//Returns or creates the appropriate thread pool for the system
ThreadPool& getSystemThreadPool();


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
