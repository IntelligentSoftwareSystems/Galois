/** Dikstra style termination detection -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in
 * irregular programs.

 * Copyright (C) 2011, The University of Texas at Austin. All rights
 * reserved.  UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES
 * CONCERNING THIS SOFTWARE AND DOCUMENTATION, INCLUDING ANY
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY PARTICULAR PURPOSE,
 * NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY WARRANTY
 * THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF
 * TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO
 * THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect,
 * direct or consequential damages or loss of profits, interruption of
 * business, or related expenses which may arise from use of Software
 * or Documentation, including but not limited to those resulting from
 * defects in Software and/or Documentation, or loss or inaccuracy of
 * data of any kind.
 *
 * @section Description
 *
 * Implementation of Termination Detection
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_TERMINATION_H
#define GALOIS_RUNTIME_TERMINATION_H

#include "Galois/config.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/ll/CacheLineStorage.h"

#include GALOIS_CXX11_STD_HEADER(atomic)

namespace Galois {
namespace Runtime {

class TerminationDetection {
protected:
  LL::CacheLineStorage<std::atomic<int> > globalTerm;
public:
  /**
   * Initializes the per-thread state.  All threads must call this
   * before any call localTermination.
   */
  virtual void initializeThread() = 0;

  /**
   * Process termination locally.  May be called as often as needed.  The
   * argument workHappened signals that since last time it was called, some
   * progress was made that should prevent termination. All threads must call
   * initializeThread() before any thread calls this function.  This function
   * should not be on the fast path (this is why it takes a flag, to allow the
   * caller to buffer up work status changes).
   */
  virtual void localTermination(bool workHappened) = 0;

  /**
   * Returns whether global termination is detected.
   */
  bool globalTermination() const {
    return globalTerm.data;
  }
};

//returns an object.  The object will be reused.
TerminationDetection& getSystemTermination();

} // end namespace Runtime
} // end namespace Galois

#endif
