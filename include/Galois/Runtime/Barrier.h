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

#include "PerThreadStorage.h"

#include <pthread.h>

namespace GaloisRuntime {

class PthreadBarrier {
  pthread_barrier_t bar;
  void checkResults(int val);

public:
  PthreadBarrier();
  PthreadBarrier(unsigned int val);
  ~PthreadBarrier();

  void reinit(int val);
  void wait();
  void operator()(void) { wait(); }
};

//! Simple busy waiting barrier, not cyclic
class SimpleBarrier {
  struct PLD {
    int count;
    int total;
    PLD(): count(0) { }
  };

  struct TLD {
    volatile int flag;
    TLD(): flag(0) { }
  };

  volatile int globalTotal;
  PerThreadStorage<TLD> tlds;
  PerPackageStorage<PLD> plds;
  int size;

  void cascade(int tid);

public:
  SimpleBarrier(): globalTotal(0), size(-1) { }

  //! Not thread-safe and should only be called when no threads are in wait()/increment()
  void reinit(int val, int init = 0);

  void increment();
  void wait();
  void barrier();
};

//! Busy waiting barrier biased towards getting master thread out as soon
//! as possible. Cyclic.
class FastBarrier {
  SimpleBarrier in;
  SimpleBarrier out;
  int size;

public:
  FastBarrier(): size(-1) { }
  explicit FastBarrier(unsigned int val): size(-1) { reinit(val); }

  void reinit(int val);
  void wait(); 
  void operator()(void) { wait(); }
};

class FasterBarrier {
  volatile unsigned count;
  unsigned P;
  bool sense;
  PerThreadStorage<volatile bool> local_sense;

public:
  FasterBarrier() :count(~0), P(~0), sense(true) {}
  explicit FasterBarrier(unsigned int val): count(~0), P(~0), sense(true) {
    reinit(val);
  }

  void reinit(unsigned num) {
    P = count = num;
  }

  void wait();
  void operator()(void) { wait(); }
};

class MCSBarrier {
  struct treenode {
    //vpid is LL::getTID()
    volatile bool* parentpointer; //null of vpid == 0
    volatile bool* childpointers[2];
    bool havechild[4];

    volatile bool childnotready[4];
    volatile bool parentsense;
    bool sense;
  };

  PerThreadStorage<treenode> nodes;
  
  void _reinit(unsigned P);

public:
  MCSBarrier();
  explicit MCSBarrier(unsigned val);

  //not safe if any thread is in wait
  void reinit(unsigned val);

  void wait();

  void operator()(void) { wait(); }
};

class TopoBarrier {
  struct treenode {
    //vpid is LL::getTID()

    //package binary tree
    treenode* parentpointer; //null of vpid == 0
    treenode* childpointers[2];

    //waiting values:
    unsigned havechild;
    volatile unsigned childnotready;

    //signal values
    volatile unsigned parentsense;
  };

  PerPackageStorage<treenode> nodes;
  PerThreadStorage<unsigned> sense;

  void _reinit(unsigned P);

public:
  TopoBarrier();
  explicit TopoBarrier(unsigned val);

  //not safe if any thread is in wait
  void reinit(unsigned val);

  void wait();

  void operator()(void) { wait(); }

  //  void dump();
};

typedef TopoBarrier GBarrier;

//! Have a pre-instantiated barrier available for use
GBarrier& getSystemBarrier();

}

#endif
