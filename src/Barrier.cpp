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
 * Barriers
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/ActiveThreads.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"
#include "Galois/Runtime/ll/gio.h"
#include <pthread.h>

#include <cstdlib>
#include <cstdio>

#if _POSIX_BARRIERS > 0
class PthreadBarrier: public Galois::Runtime::Barrier {
  pthread_barrier_t bar;

public:
  PthreadBarrier() {
    if (pthread_barrier_init(&bar, 0, ~0))
      GALOIS_DIE("PTHREAD");
  }
  
  PthreadBarrier(unsigned int val) {
    if (pthread_barrier_init(&bar, 0, val))
      GALOIS_DIE("PTHREAD");
  }

  virtual ~PthreadBarrier() {
    if (pthread_barrier_destroy(&bar))
      GALOIS_DIE("PTHREAD");
  }

  virtual void reinit(unsigned val) {
    if (pthread_barrier_destroy(&bar))
      GALOIS_DIE("PTHREAD");
    if (pthread_barrier_init(&bar, 0, val))
      GALOIS_DIE("PTHREAD");
  }

  virtual void wait() {
    int rc = pthread_barrier_wait(&bar);
    if (rc && rc != PTHREAD_BARRIER_SERIAL_THREAD)
      GALOIS_DIE("PTHREAD");
  }
};
#endif

class OneWayBarrier: public Galois::Runtime::Barrier {
  pthread_mutex_t lock;
  pthread_cond_t cond;
  unsigned count; 
  unsigned total;

public:
  OneWayBarrier(unsigned p = Galois::Runtime::activeThreads) {
    if (pthread_mutex_init(&lock, NULL))
      GALOIS_DIE("PTHREAD");
    if (pthread_cond_init(&cond, NULL))
      GALOIS_DIE("PTHREAD");
    reinit(p);
  }
  
  virtual ~OneWayBarrier() {
    if (pthread_mutex_destroy(&lock))
      GALOIS_DIE("PTHREAD");
    if (pthread_cond_destroy(&cond))
      GALOIS_DIE("PTHREAD");
  }

  virtual void reinit(unsigned val) {
    count = 0;
    total = val;
  }

  virtual void wait() {
    if (pthread_mutex_lock(&lock))
      GALOIS_DIE("PTHREAD");
    count += 1;
    while (count < total) {
      if (pthread_cond_wait(&cond, &lock))
        GALOIS_DIE("PTHREAD");
    }
    if (pthread_cond_broadcast(&cond))
        GALOIS_DIE("PTHREAD");

    if (pthread_mutex_unlock(&lock))
      GALOIS_DIE("PTHREAD");
  }
};

class SimpleBarrier: public Galois::Runtime::Barrier {
  OneWayBarrier barrier1;
  OneWayBarrier barrier2;
  unsigned total;
public:
  SimpleBarrier(unsigned p = Galois::Runtime::activeThreads): barrier1(p), barrier2(p), total(p) { }

  virtual ~SimpleBarrier() { }

  virtual void reinit(unsigned val) {
    total = val;
    barrier1.reinit(val);
    barrier2.reinit(val);
  }

  virtual void wait() {
    barrier1.wait();
    if (Galois::Runtime::LL::getTID() == 0)
      barrier1.reinit(total);
    barrier2.wait();
    if (Galois::Runtime::LL::getTID() == 0)
      barrier2.reinit(total);
  }

};

class MCSBarrier: public Galois::Runtime::Barrier {
  struct treenode {
    //vpid is Galois::Runtime::LL::getTID()
    volatile bool* parentpointer; //null of vpid == 0
    volatile bool* childpointers[2];
    bool havechild[4];

    volatile bool childnotready[4];
    volatile bool parentsense;
    bool sense;
  };

  Galois::Runtime::PerThreadStorage<treenode> nodes;
  
  void _reinit(unsigned P) {
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

public:
  MCSBarrier(unsigned P = Galois::Runtime::activeThreads) {
    _reinit(P);
  }

  virtual void reinit(unsigned val) {
    _reinit(val);
  }

  virtual void wait() {
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
};

class TopoBarrier : public Galois::Runtime::Barrier {
  struct treenode {
    //vpid is Galois::Runtime::LL::getTID()

    //package binary tree
    treenode* parentpointer; //null of vpid == 0
    treenode* childpointers[2];

    //waiting values:
    unsigned havechild;
    volatile unsigned childnotready;

    //signal values
    volatile unsigned parentsense;
  };

  Galois::Runtime::PerPackageStorage<treenode> nodes;
  Galois::Runtime::PerThreadStorage<unsigned> sense;

  void _reinit(unsigned P) {
    unsigned pkgs = Galois::Runtime::LL::getMaxPackageForThread(P-1) + 1;
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
      for (unsigned j = 0; j < P; ++j) {
	if (Galois::Runtime::LL::getPackageForThread(j) == i && !Galois::Runtime::LL::isPackageLeader(j)) {
          ++n.childnotready;
          ++n.havechild;
        }
      }
      n.parentpointer = (i == 0) ? 0 : nodes.getRemoteByPkg((i-1)/4);
      n.childpointers[0] = ((2*i + 1) >= pkgs) ? 0 : nodes.getRemoteByPkg(2*i+1);
      n.childpointers[1] = ((2*i + 2) >= pkgs) ? 0 : nodes.getRemoteByPkg(2*i+2);
      n.parentsense = 0;
    }
    for (unsigned i = 0; i < P; ++i)
      *sense.getRemote(i) = 1;
#if 0
    for (unsigned i = 0; i < pkgs; ++i) {
      treenode& n = *nodes.getRemoteByPkg(i);
      Galois::Runtime::LL::gPrint(i, 
          " this ", &n,
          " parent ", n.parentpointer, 
          " child[0] ", n.childpointers[0],
          " child[1] ", n.childpointers[1],
          " havechild ", n.havechild,
          "\n");
    }
#endif
  }

public:
  TopoBarrier(unsigned val = Galois::Runtime::activeThreads) {
    _reinit(val);
  }

  //not safe if any thread is in wait
  virtual void reinit(unsigned val) {
    _reinit(val);
  }

  virtual void wait() {
    unsigned id = Galois::Runtime::LL::getTID();
    treenode& n = *nodes.getLocal();
    unsigned& s = *sense.getLocal();
    bool leader = Galois::Runtime::LL::isPackageLeaderForSelf(id);
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
      while (n.parentsense != s) {
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
};

Galois::Runtime::Barrier::~Barrier() {}

Galois::Runtime::Barrier* Galois::Runtime::createSimpleBarrier() {
#if _POSIX_BARRIERS > 1
  return new PthreadBarrier();
#else
  return new SimpleBarrier();
#endif
}

Galois::Runtime::Barrier& Galois::Runtime::getSystemBarrier() {
  static TopoBarrier b;
  static unsigned num = ~0;
  if (activeThreads != num) {
    num = activeThreads;
    b.reinit(num);
  }
  return b;
}

