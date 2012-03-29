#include "Galois/Runtime/Barrier.h"
#include "Galois/Runtime/Config.h"

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
    GaloisRuntime::Config::compilerBarrier();
    __sync_add_and_fetch(&globalTotal, total);
  }
}

void GaloisRuntime::SimpleBarrier::wait() {
  int tid = (int) LL::getTID();
  TLD& tld = tlds.get(tid);
  if (tid == 0) {
    while (globalTotal < size) {
      GaloisRuntime::Config::pause();
    }
  } else {
    while (!tld.flag) {
      GaloisRuntime::Config::pause();
    }
  }
}

void GaloisRuntime::SimpleBarrier::barrier() {
  assert(size > 0);

  int tid = (int) LL::getTID();
  TLD& tld = tlds.get(tid);

  if (tid == 0) {
    while (globalTotal < size) {
      GaloisRuntime::Config::pause();
    }

    globalTotal = 0;
    tld.flag = 0;
    GaloisRuntime::Config::compilerBarrier();
    cascade(tid);
  } else {
    while (!tld.flag) {
      GaloisRuntime::Config::pause();
    }

    tld.flag = 0;
    GaloisRuntime::Config::compilerBarrier();
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


