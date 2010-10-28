// -*- C++ -*-
/*! \file 
 *  \brief simple thread pool base classes
 */

namespace GaloisRuntime {
  
  class Executable {
  public:
    //! run work.  id is the thread id, max is the total num
    virtual void operator()(int id, int tmax) = 0;
  };

  class ThreadPool {

  public:
    //!execute work on all threads
    //!The work object is not duplicated in any way 
    virtual void run(Executable* work) = 0;

    //!change the number of preallocated threads
    virtual void resize(int num) = 0;

    //!How many threads are kept around
    virtual int size() = 0;
  };

  //Returns or creates the appropriate thread pool for the system
  ThreadPool& getSystemThreadPool();
  
}
