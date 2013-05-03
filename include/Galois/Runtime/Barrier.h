/** Barriers -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_BARRIER_H
#define GALOIS_RUNTIME_BARRIER_H

namespace Galois {
namespace Runtime {

class Barrier {
public:
  virtual ~Barrier();

  //not safe if any thread is in wait
  virtual void reinit(unsigned val) = 0;

  //Wait at this barrier
  virtual void wait() = 0;

  //wait at this barrier
  void operator()(void) { wait(); }
};

/**
 * Have a pre-instantiated barrier available for use.
 * This is initialized to the current activeThreads. This barrier
 * is designed to be fast and should be used in the common
 * case. 
 *
 * However, there is a race if the number of active threads
 * is modified after using this barrier: some threads may still
 * be in the barrier while the main thread reinitializes this
 * barrier to the new number of active threads. If that may
 * happen, use {@link createSimpleBarrier} instead. 
 */
Barrier& getSystemBarrier();

/**
 * Creates a new simple barrier. This barrier is not designed to be fast but
 * does gaurantee that all threads have left the barrier before returning
 * control. Useful when the number of active threads is modified to avoid a
 * race in {@link getSystemBarrier}.  Client is reponsible for deallocating
 * returned barrier.
 */
Barrier* createSimpleBarrier();
}
} // end namespace Galois

#endif
