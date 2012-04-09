/** Galois configuration -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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

void GaloisRuntime::SimpleBarrier::cascade(int tid) {
  int multiple = 2;
  for (int i = 0; i < multiple; ++i) {
    int n = tid * multiple + i;
    if (n < size && n != 0)
      tlds.get(n).flag = 1;
  }
}

void GaloisRuntime::SimpleBarrier::reinit(int val, int init) {
  assert(val > 0);

  if (val != size) {
    for (unsigned i = 0; i < plds.size(); ++i)
      plds.get(i).total = 0;
    for (int i = 0; i < val; ++i) {
      int j = LL::getPackageForThreadInternal(i);
      ++plds.get(j).total;
    }

    size = val;
  }

  globalTotal = init;
}

void GaloisRuntime::SimpleBarrier::increment() {
  PLD& pld = plds.get();
  int total = pld.total;

  if (__sync_add_and_fetch(&pld.count, 1) == total) {
    pld.count = 0;
    GaloisRuntime::LL::compilerBarrier();
    __sync_add_and_fetch(&globalTotal, total);
  }
}

void GaloisRuntime::SimpleBarrier::wait() {
  int tid = (int) LL::getTID();
  TLD& tld = tlds.get(tid);
  if (tid == 0) {
    while (globalTotal < size) {
      GaloisRuntime::LL::mem_pause();
    }
  } else {
    while (!tld.flag) {
      GaloisRuntime::LL::mem_pause();
    }
  }
}

void GaloisRuntime::SimpleBarrier::barrier() {
  assert(size > 0);

  int tid = (int) LL::getTID();
  TLD& tld = tlds.get(tid);

  if (tid == 0) {
    while (globalTotal < size) {
      GaloisRuntime::LL::mem_pause();
    }

    globalTotal = 0;
    tld.flag = 0;
    GaloisRuntime::LL::compilerBarrier();
    cascade(tid);
  } else {
    while (!tld.flag) {
      GaloisRuntime::LL::mem_pause();
    }

    tld.flag = 0;
    GaloisRuntime::LL::compilerBarrier();
    cascade(tid);
  }

}

void GaloisRuntime::FastBarrier::reinit(int val) {
  if (val != size) {
    if (size != -1)
      out.wait();

    in.reinit(val, 0);
    out.reinit(val, val);
    val = size;
  }
}

void GaloisRuntime::FastBarrier::wait() {
  out.barrier();
  in.increment();
  in.barrier();
  out.increment();
}


