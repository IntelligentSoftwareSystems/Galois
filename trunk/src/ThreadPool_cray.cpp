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
/*! \file 
 *  \brief cray thread pool implementation
 */

#ifdef GALOIS_CRAY

#include "Galois/Runtime/ThreadPool.h"

using namespace GaloisRuntime;

namespace {

  class ThreadPool_cray : public ThreadPool {
    
    int tmax;
    int num;
    Executable* work;

    int mkID() {
      return int_fetch_add(&tmax, 1);
    }

    void launch(void) {
      int myID = mkID();
      (*work)(myID, num);
    }

  public:
    ThreadPool_cray() 
      :tmax(0), num(1)
    {}

    virtual void run(Executable* E) {
      work = E;
      work->preRun(num);
#pragma mta assert parallel
      for (int i = 0; i < num; ++i) {
	launch();
      }
      work->postRun();
    }

    virtual void resize(int num) {
      this->num = num;
    }

    virtual int size() {
      return num;
    }
  };
}


//! Implement the global threadpool
static ThreadPool_cray pool;

ThreadPool& GaloisRuntime::getSystemThreadPool() {
  return pool;
}

#endif
