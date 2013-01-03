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
 */

#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

#include <cstdlib>
#include <cstdio>

#ifdef GALOIS_USE_DMP
#include "dmp.h"
#endif

void Galois::Runtime::PthreadBarrier::checkResults(int val) {
  if (val) {
    perror("PTHREADS: ");
    assert(0 && "PThread check");
    abort();
  }
}

Galois::Runtime::PthreadBarrier::PthreadBarrier() {
  //uninitialized barriers block a lot of threads to help with debugging
  int rc = pthread_barrier_init(&bar, 0, ~0);
  checkResults(rc);
}

Galois::Runtime::PthreadBarrier::PthreadBarrier(unsigned int val) {
  int rc = pthread_barrier_init(&bar, 0, val);
  checkResults(rc);
}

Galois::Runtime::PthreadBarrier::~PthreadBarrier() {
  int rc = pthread_barrier_destroy(&bar);
  checkResults(rc);
}

void Galois::Runtime::PthreadBarrier::reinit(int val) {
  int rc = pthread_barrier_destroy(&bar);
  checkResults(rc);
  rc = pthread_barrier_init(&bar, 0, val);
  checkResults(rc);
}

void Galois::Runtime::PthreadBarrier::wait() {
  int rc = pthread_barrier_wait(&bar);
  if (rc && rc != PTHREAD_BARRIER_SERIAL_THREAD)
    checkResults(rc);
}

void Galois::Runtime::SimpleBarrier::cascade(int tid) {
  int multiple = 2;
  for (int i = 0; i < multiple; ++i) {
    int n = tid * multiple + i;
    if (n < size && n != 0)
      tlds.getRemote(n)->flag = 1;
  }
}

void Galois::Runtime::SimpleBarrier::reinit(int val, int init) {
  assert(val > 0);

  if (val != size) {
    for (unsigned i = 0; i < plds.size(); ++i)
      plds.getRemote(i)->total = 0;
    for (int i = 0; i < val; ++i) {
      int j = LL::getPackageForThread(i);
      ++plds.getRemote(j)->total;
    }

    size = val;
  }

  globalTotal = init;
}

void Galois::Runtime::SimpleBarrier::increment() {
  PLD& pld = *plds.getLocal();
  int total = pld.total;

  if (__sync_add_and_fetch(&pld.count, 1) == total) {
    pld.count = 0;
    Galois::Runtime::LL::compilerBarrier();
    __sync_add_and_fetch(&globalTotal, total);
  }
}

void Galois::Runtime::SimpleBarrier::wait() {
  int tid = (int) LL::getTID();
  TLD& tld = *tlds.getLocal();
  if (tid == 0) {
    while (globalTotal < size) {
      Galois::Runtime::LL::asmPause();
    }
  } else {
    while (!tld.flag) {
      Galois::Runtime::LL::asmPause();
    }
  }
}

void Galois::Runtime::SimpleBarrier::barrier() {
  assert(size > 0);

  int tid = (int) LL::getTID();
  TLD& tld = *tlds.getLocal();

  if (tid == 0) {
    while (globalTotal < size) {
      Galois::Runtime::LL::asmPause();
    }

    globalTotal = 0;
    tld.flag = 0;
    Galois::Runtime::LL::compilerBarrier();
    cascade(tid);
  } else {
    while (!tld.flag) {
      Galois::Runtime::LL::asmPause();
    }

    tld.flag = 0;
    Galois::Runtime::LL::compilerBarrier();
    cascade(tid);
  }

}

void Galois::Runtime::FastBarrier::reinit(int val) {
  if (val != size) {
    if (size != -1)
      out.wait();

    in.reinit(val, 0);
    out.reinit(val, val);
    val = size;
  }
}

void Galois::Runtime::FastBarrier::wait() {
  out.barrier();
  in.increment();
  in.barrier();
  out.increment();
}

#define fanout 4
void Galois::Runtime::FasterBarrier::wait() {
  volatile bool& ls = *local_sense.getLocal();
  if (0 == __sync_sub_and_fetch(&count, 1)) {
    count = P;
    LL::compilerBarrier();
    *local_sense.getRemote(0) = true;
  }
  while (!ls) { LL::asmPause(); }
  ls = false;
  unsigned id = LL::getTID();
  for (unsigned x = 1; x <= fanout; ++x)
    if (id * fanout + x < P)
      *local_sense.getRemote(id*fanout+x) = true;
}

void Galois::Runtime::MCSBarrier::_reinit(unsigned P) {
  for (unsigned i = 0; i < nodes.size(); ++i) {
    treenode& n = *nodes.getRemote(i);
    n.sense = true;
    n.parentsense = false;
    for (int j = 0; j < 4; ++j)
      n.childnotready[j] = n.havechild[j] = ((4*i+j+1) < P);
    n.parentpointer = (i == 0) ? 0 :
      &nodes.getRemote((i-1)/4)->childnotready[(i-1)%4];
    n.childpointers[0] = ((2*i + 1) >= P) ? 0 :
      &nodes.getRemote(2*i+1)->parentsense;
    n.childpointers[1] = ((2*i + 2) >= P) ? 0 :
      &nodes.getRemote(2*i+2)->parentsense;
  }
}

Galois::Runtime::MCSBarrier::MCSBarrier() {
  _reinit(galoisActiveThreads);
}

Galois::Runtime::MCSBarrier::MCSBarrier(unsigned P) {
  _reinit(P);
}

void Galois::Runtime::MCSBarrier::reinit(unsigned val) {
  _reinit(val);
}

void Galois::Runtime::MCSBarrier::wait() {
  treenode& n = *nodes.getLocal();
  while (n.childnotready[0] || n.childnotready[1] || 
   	 n.childnotready[2] || n.childnotready[3]) {
    Galois::Runtime::LL::asmPause();
  }
  for (int i = 0; i < 4; ++i)
    n.childnotready[i] = n.havechild[i];
  if (n.parentpointer) {
    //FIXME: make sure the compiler doesn't do a RMW because of the as-if rule
    *n.parentpointer = false;
    while(n.parentsense != n.sense) {
      Galois::Runtime::LL::asmPause();
    }
  }
  //signal children in wakeup tree
  if (n.childpointers[0])
    *n.childpointers[0] = n.sense;
  if (n.childpointers[1])
    *n.childpointers[1] = n.sense;
  n.sense = !n.sense;
}









void Galois::Runtime::TopoBarrier::_reinit(unsigned P) {
  unsigned pkgs = LL::getMaxPackageForThread(P-1) + 1;
  for (unsigned i = 0; i < pkgs; ++i) {
    treenode& n = *nodes.getRemoteByPkg(i);
    n.childnotready = 0;
    n.havechild = 0;
    for (int j = 0; j < 4; ++j) {
	if ((4*i+j+1) < pkgs) {
	  ++n.childnotready;
	  ++n.havechild;
	}
    }
    for (unsigned j = 0; j < P; ++j)
      if (LL::getPackageForThread(j) == i && !LL::isPackageLeader(j)) {
	++n.childnotready;
	++n.havechild;
      }
    n.parentpointer = (i == 0) ? 0 : nodes.getRemoteByPkg((i-1)/4);
    n.childpointers[0] = ((2*i + 1) >= pkgs) ? 0 : nodes.getRemoteByPkg(2*i+1);
    n.childpointers[1] = ((2*i + 2) >= pkgs) ? 0 : nodes.getRemoteByPkg(2*i+2);
    n.parentsense = 0;
  }
  for (unsigned i = 0; i < P; ++i)
    *sense.getRemote(i) = 1;
}

Galois::Runtime::TopoBarrier::TopoBarrier() {
  _reinit(galoisActiveThreads);
}

Galois::Runtime::TopoBarrier::TopoBarrier(unsigned P) {
  _reinit(P);
}

void Galois::Runtime::TopoBarrier::reinit(unsigned val) {
  _reinit(val);
}

void Galois::Runtime::TopoBarrier::wait() {
  unsigned id = LL::getTID();
  treenode& n = *nodes.getLocal();
  unsigned& s = *sense.getLocal();
  bool leader = LL::isPackageLeaderForSelf(id);
  //completion tree
  if (leader) {
    while (n.childnotready) { Galois::Runtime::LL::asmPause(); }
    n.childnotready = n.havechild;
    if (n.parentpointer) {
      __sync_fetch_and_sub(&n.parentpointer->childnotready, 1);
    }
  } else {
    __sync_fetch_and_sub(&n.childnotready, 1);
  }

  //wait for signal
  if (id != 0) {
    while(n.parentsense != s) {
      Galois::Runtime::LL::asmPause();
    }
  }

  //signal children in wakeup tree
  if (leader) {
    if (n.childpointers[0])
      n.childpointers[0]->parentsense = s;
    if (n.childpointers[1])
      n.childpointers[1]->parentsense = s;
    if (id == 0)
      n.parentsense = s;
  }
  ++s;
}

// void Galois::Runtime::TopoBarrier::dump() {
//   unsigned pkgs = LL::getMaxPackages();
//   for (unsigned i = 0; i < pkgs; ++i) {
//     treenode* n = nodes.getRemoteByPkg(i);
//     std::cerr << n << " " << n->parentpointer << " " << n->childpointers[0] << " " << n->childpointers[1] << " " << n->havechild << " " << n->childnotready << " " << n->parentsense << "\n";
//   }
//   for (unsigned i = 0; i < sense.size(); ++i) {
//     std::cerr << *sense.getRemote(i) << " ";
//   }
//   std::cerr << "\n";

// }

Galois::Runtime::GBarrier& Galois::Runtime::getSystemBarrier() {
  static Galois::Runtime::GBarrier b;
  static unsigned num = ~0;
  if (galoisActiveThreads != num) {
    num = galoisActiveThreads;
    b.reinit(num);
  }
  return b;
}
