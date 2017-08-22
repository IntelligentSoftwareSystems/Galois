/** Dikstra style termination detection -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Implementation of Termination Detection
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_SUBSTRATE_TERMINATION_H
#define GALOIS_SUBSTRATE_TERMINATION_H

#include "Galois/Substrate/PerThreadStorage.h"
#include "Galois/Substrate/CacheLineStorage.h"

#include <atomic>

namespace Galois {
namespace Substrate {

class TerminationDetection {
protected:
  CacheLineStorage<std::atomic<int> > globalTerm;
public:
  virtual ~TerminationDetection(void);
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

//returns an object.  The object will be reused, but reinitialized to activeThreads
TerminationDetection& getSystemTermination(unsigned activeThreads);

namespace internal {
  void initTermDetect(void);
  void killTermDetect(void);
}

} // end namespace Runtime
} // end namespace Galois

#endif
