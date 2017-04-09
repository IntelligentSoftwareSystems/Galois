/** Galois barrier -*- C++ -*-
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
 * @section Description
 *
 * Fast Barrier
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#include "Galois/Substrate/PerThreadStorage.h"
#include "Galois/Runtime/Substrate.h"
#include "Galois/Substrate/CompilerSpecific.h"
#include "Galois/Runtime/Network.h"
//#include "Galois/Runtime/Directory.h"

#include <cstdlib>
#include <cstdio>
/*
class StupidDistBarrier : public Galois::Substrate::Barrier {
  Barrier& localBarrier;

  std::atomic<int> count;

  static void barrierLandingPad() {
    //std::cout << "inside barrierLandingpad , count : " << getDistBarrier().count << "\n";
    --getDistBarrier().count;
  }

public:
  
  StupidDistBarrier() : count(0) {}
  
  virtual const char* name() const { return "DistBarrier"; }

  virtual void reinit(unsigned val) {
    localBarrier.reinit(val);
  }
  
  virtual void wait() {
    if (Galois::Runtime::LL::getTID() == 0) {
      count += Galois::Runtime::NetworkInterface::Num;
    }
    
    //wait at local barrier
    localBarrier.wait();

    auto& net = Galois::Runtime::getSystemNetworkInterface();
    if (Galois::Runtime::LL::getTID() == 0) {
      //notify global and wait on global
      net.broadcastAlt(barrierLandingPad);
      --count;
    }
    
    while (count > 0) {
      Galois::Runtime::doNetworkWork();
    }
    
    //wait at local barrier
    localBarrier.wait();
  }
};
*/

#include <iostream>
#include <mpi.h>

#ifdef GALOIS_USE_LWCI
void __fence__();
#endif

namespace {
class HostBarrier : public Galois::Substrate::Barrier {
  std::atomic<int> count;

  static void barrierLandingPad(uint32_t) {
    --static_cast<HostBarrier&>(Galois::Runtime::getHostBarrier()).count;
  }

public:
  HostBarrier() : count(0) {}
  
  virtual const char* name() const { return "HostBarrier"; }

  virtual void reinit(unsigned val) { }
  
  virtual void wait() {
#ifdef GALOIS_USE_LWCI
    __fence__();
#endif

    if (Galois::Substrate::ThreadPool::getTID() == 0) {
      count += Galois::Runtime::NetworkInterface::Num;
    }
    
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    if (Galois::Substrate::ThreadPool::getTID() == 0) {
      //      std::cerr << "@";
      //notify global and wait on global
      net.broadcastSimple(barrierLandingPad);
      --count;
    }
    //    std::cerr << "#";
    
    while (count > 0) {
      net.handleReceives();
    }
    //    std::cerr << "$";
  }
};
} // end namespace ""

Galois::Substrate::Barrier& Galois::Runtime::getHostBarrier() {
  static HostBarrier b;
  return b;
}

