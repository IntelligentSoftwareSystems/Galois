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
#ifndef GALOIS_RUNTIME_THREADPOOL_H
#define GALOIS_RUNTIME_THREADPOOL_H

#include "Galois/Runtime/Config.h"

namespace GaloisRuntime {

//! the number of active threads
extern unsigned int galoisActiveThreads;

typedef Config::function<void (void)> RunCommand;

class ThreadPool {
public:
  virtual ~ThreadPool() { }

  //!execute work on all threads
  //!preWork and postWork are executed only on the master thread
  virtual void run(RunCommand* begin, RunCommand* end, unsigned num = galoisActiveThreads) = 0;

  //!return the number of threads supported by the thread pool on the current machine
  virtual unsigned getMaxThreads() const = 0;

};

//!Returns or creates the appropriate thread pool for the system
ThreadPool& getSystemThreadPool();

}

#endif
