/** Basic Galois Runtime -*- C++ -*-
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
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_ENVCHECK_H
#define GALOIS_RUNTIME_ENVCHECK_H

#include <ThreadPool.h>

namespace Galois {
namespace Runtime {

class GaloisRuntime {
  ThreadPool& threadpool;
  std::unique_ptr<TerminationDetection> term;

public:
  GaloisRuntime()
    :threadpool(ThreadPool::getThreadPool()),
     term(getTermination())
  {
    //initialize PTS
    PerThreadStorage<int> foo;
  }

  ThreadPool& getThreadPool() { return threadpool; }
  
  TerminationDetection& getTermination(unsigned activeThreads) {
    term->init(activeThreads);
    return *term;
  }

}

} // end namespace Substrate
} // end namespace Galois

#endif
