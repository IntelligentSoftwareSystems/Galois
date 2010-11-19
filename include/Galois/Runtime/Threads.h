// -*- C++ -*-
/*! \file 
 *  \brief simple thread related classes
 */

#ifndef _GALOISRUNTIME_THREADS_H
#define _GALOISRUNTIME_THREADS_H

#include "Galois/Executable.h"

#include <boost/intrusive/list.hpp>

namespace GaloisRuntime {

class ThreadPool {
protected:
  static void NotifyAware(int n);
  static void ResetThreadNumbers();

public:
  //!execute work on all threads
  //!The work object is not duplicated in any way 
  virtual void run(Galois::Executable* work) = 0;
  
  //!change the number of preallocated threads
  virtual void resize(int num) = 0;
  
  //!How many threads are kept around
  virtual int size() = 0;

  static int getMyID();
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
  static void NotifyOfChange(int num);

protected:
  void init();

public:
  ThreadAware();
  virtual ~ThreadAware();

  //This is called to notify changes in the number of threads
  //Thread 0 always exists (inital thread).
  //Parallel code has threads labeled from [1 -> num]
  //num is zero when Parallel code is exiting
  virtual void ThreadChange(int num) = 0;

};

}

#endif
