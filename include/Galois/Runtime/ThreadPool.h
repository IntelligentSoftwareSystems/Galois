// -*- C++ -*-
/*! \file 
 *  \brief simple thread pool base classes
 */

#ifndef _GALOISRUNTIME_THREADPOOL_H
#define _GALOISRUNTIME_THREADPOOL_H

#include "Galois/Executable.h"

namespace GaloisRuntime {
  
  class ThreadPool {

  public:
    //!execute work on all threads
    //!The work object is not duplicated in any way 
    virtual void run(Galois::Executable* work) = 0;

    //!change the number of preallocated threads
    virtual void resize(int num) = 0;

    //!How many threads are kept around
    virtual int size() = 0;
  };

  //Returns or creates the appropriate thread pool for the system
  ThreadPool& getSystemThreadPool();
  
}

#endif
