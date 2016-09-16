/** Galois Simple Function Executor -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
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
 * @section Description
 *
 * Simple wrapper for the thread pool
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_EXECUTOR_ONEACH_H
#define GALOIS_RUNTIME_EXECUTOR_ONEACH_H

#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/Sampling.h"
#include "Galois/Runtime/Statistics.h"

namespace Galois {
namespace Runtime {

template<typename FunctionTy>
void on_each_impl(unsigned activeThreads, const FunctionTy& fn, const char* loopname = nullptr) {
  reportLoopInstance(loopname);
  beginSampling(loopname);
  ThreadPool::getThreadPool().run(activeThreads,
                                  [&fn, activeThreads] () {
                                    fn(ThreadPool::getTID(), activeThreads);
                                  });
  endSampling();
}

} // end namespace Runtime
} // end namespace Galois

#endif
