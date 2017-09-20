/** Galois barrier -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#include "Galois/Substrate/Barrier.h"
#include "Galois/Substrate/ThreadPool.h"

#include <mutex>
#include <condition_variable>

namespace {

class OneWayBarrier: public galois::Substrate::Barrier {
  std::mutex lock;
  std::condition_variable cond;
  unsigned count; 
  unsigned total;

public:
  OneWayBarrier(unsigned p) {
    reinit(p);
  }
  
  virtual ~OneWayBarrier() {
  }

  virtual void reinit(unsigned val) {
    count = 0;
    total = val;
  }

  virtual void wait() {
    std::unique_lock<std::mutex> tmp(lock);
    count += 1;
    cond.wait(tmp, [this] () { return count >= total; });
    cond.notify_all();
  }

  virtual const char* name() const { return "OneWayBarrier"; }
};

class SimpleBarrier: public galois::Substrate::Barrier {
  OneWayBarrier barrier1;
  OneWayBarrier barrier2;
  unsigned total;
public:
  SimpleBarrier(unsigned p): barrier1(p), barrier2(p), total(p) { }

  virtual ~SimpleBarrier() { }

  virtual void reinit(unsigned val) {
    total = val;
    barrier1.reinit(val);
    barrier2.reinit(val);
  }

  virtual void wait() {
    barrier1.wait();
    if (galois::Substrate::ThreadPool::getTID() == 0)
      barrier1.reinit(total);
    barrier2.wait();
    if (galois::Substrate::ThreadPool::getTID() == 0)
      barrier2.reinit(total);
  }

  virtual const char* name() const { return "SimpleBarrier"; }

};

} // end anonymous namespace

std::unique_ptr<galois::Substrate::Barrier> galois::Substrate::createSimpleBarrier(unsigned int v) {
  return std::unique_ptr<Barrier>(new SimpleBarrier(v));
}

