/** Simple thread related classes -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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

#include "Galois/config.h"
#include GALOIS_CXX11_STD_HEADER(functional)

namespace Galois {
namespace Runtime {

typedef std::function<void (void)> RunCommand;

class ThreadPool {
protected:
  unsigned maxThreads;
  ThreadPool(unsigned m) :maxThreads(m) {}
public:
  virtual ~ThreadPool() { }

  //!execute work on all threads
  //!preWork and postWork are executed only on the master thread
  virtual void run(RunCommand* begin, RunCommand* end, unsigned num) = 0;

  //!return the number of threads supported by the thread pool on the current machine
  unsigned getMaxThreads() const { return maxThreads; }
};

//!Returns or creates the appropriate thread pool for the system
ThreadPool& getSystemThreadPool();

} //Runtime
} //Galois

#endif
