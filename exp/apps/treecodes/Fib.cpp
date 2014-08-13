#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <sstream>


#include "Galois/Galois.h"
#include "Galois/CilkInit.h"
#include "Galois/GaloisUnsafe.h"
#include "Galois/Atomic.h"
#include "Galois/Statistic.h"
#include "Galois/Runtime/DoAllCoupled.h"
#include "Galois/Runtime/Sampling.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"
#include "Galois/Runtime/TreeExec.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

namespace cll = llvm::cl;
static cll::opt<unsigned> N("n", cll::desc("n-th fibonacci number"), cll::init(39));

enum ExecType {
  SERIAL, CILK, GALOIS, GALOIS_ALT, GALOIS_STACK, GALOIS_GENERIC,
};

static cll::opt<ExecType> execType (
    cll::desc ("executor type"),
    cll::values (
      clEnumVal (SERIAL, "serial recursive"),
      clEnumVal (CILK, "CILK divide and conquer implementation"),
      clEnumVal (GALOIS, "galois divide and conquer implementation"),
      clEnumVal (GALOIS_ALT, "galois alternate divide and conquer implementation"),
      clEnumVal (GALOIS_STACK, "galois using thread stack"),
      clEnumVal (GALOIS_GENERIC, "galois std::function version"),
      clEnumValEnd),

    cll::init (SERIAL));

const char* name = "fib";
const char* desc = "compute n-th fibonacci number";
const char* url = "fib";

unsigned fib(unsigned n)
{
  if (n <= 2)
    return n;
  unsigned x = cilk_spawn fib(n-1);
  // unsigned y = fib(n-2);
  unsigned y = cilk_spawn fib(n-2);
  cilk_sync;
  return x + y;
}

unsigned serialFib (unsigned n) {
  if (n <= 2) { 
    return n;
  }

  return serialFib (n-1) + serialFib (n-2);
}


struct FibEntry {
  unsigned n;
  unsigned result;
};

struct GaloisDivide {

  template <typename C>
  void operator () (FibEntry& x, C& wl) {
    if (x.n <= 2) {
      x.result = x.n;
      return;
    }

    FibEntry y; y.n = x.n - 1;
    FibEntry z; z.n = x.n - 2;

    wl.push (y);
    wl.push (z);
  }
};

struct GaloisConquer {

  template <typename I>
  void operator () (FibEntry& x, I beg, I end) {
    if (beg != end) {
      unsigned sum = 0;
      for (I i = beg; i != end; ++i) {
        sum += i->result;
      }

      x.result = sum;
    }
  }
};

unsigned galoisFib (unsigned n) {
  FibEntry initial {n, 0};

  FibEntry final = Galois::Runtime::for_each_ordered_tree (
      initial,
      GaloisDivide (),
      GaloisConquer (),
      Galois::Runtime::TreeExecNeedsChildren (),
      "fib-galois");

  return final.result;
}

struct FibRecord {
  unsigned n;
  unsigned* result;
  unsigned term_n_1;
  unsigned term_n_2;
};

struct GaloisDivideAlt {
  template <typename C>
  void operator () (FibRecord& r, C& wl) {
    if (r.n <= 2) {
      r.term_n_1 = r.n;
      r.term_n_2 = 0;
      return;
    }

    FibRecord left {r.n-1, &(r.term_n_1), 0, 0 };

    FibRecord rigt {r.n-2, &(r.term_n_2), 0, 0 };

    wl.push (left);
    wl.push (rigt);
  }
};

struct GaloisConquerAlt {
  void operator () (FibRecord& r) {
    *(r.result) = r.term_n_1 + r.term_n_2;
  };
};

unsigned galoisFibAlt (unsigned n) {

  unsigned result = 0;

  FibRecord init { n, &result, 0, 0};

  Galois::Runtime::for_each_ordered_tree (
      init,
      GaloisDivideAlt (),
      GaloisConquerAlt (),
      "fib-galois-alt");

  return result;

}


struct GaloisFibStack {
  unsigned n;
  unsigned result;

  template <typename C>
  void operator () (C& ctx) {
    if (n <= 2) {
      result = n;
      return;
    }

    GaloisFibStack left {n-1, 0};
    ctx.spawn (left);

    GaloisFibStack right {n-2, 0};
    ctx.spawn (right);

    ctx.sync ();

    result = left.result + right.result;
  }
};

unsigned galoisFibStack (unsigned n) {
  GaloisFibStack init {n, 0};

  Galois::Runtime::for_each_ordered_tree (init, "fib");

  return init.result;
}

struct GaloisFibGeneric: public Galois::Runtime::TreeTaskBase {
  unsigned n;
  unsigned result;

  GaloisFibGeneric (unsigned _n, unsigned _result): 
    Galois::Runtime::TreeTaskBase (),
    n (_n),
    result (_result)
  {}

  virtual void execute (void) {
    if (n <= 2) {
      result = n;
      return;
    }

    GaloisFibGeneric left {n-1, 0};
    Galois::Runtime::spawn (left);

    GaloisFibGeneric right {n-2, 0};
    Galois::Runtime::spawn (right);

    Galois::Runtime::sync ();

    result = left.result + right.result;
  }
};

unsigned galoisFibGeneric (unsigned n) {
  GaloisFibGeneric init {n, 0};

  Galois::Runtime::for_each_ordered_tree_generic (init, "fib-gen");
  return init.result;
}


int main (int argc, char* argv[]) {

  Galois::StatManager sm;
  LonestarStart (argc, argv, name, desc, url);

  unsigned result = -1;

  Galois::StatTimer t;

  t.start ();
  switch (execType) {
    case SERIAL:
      result = serialFib (N);
      break;

    case CILK:
      Galois::CilkInit ();
      result = fib (N);
      break;

    case GALOIS:
      result = galoisFib (N);
      break;

    case GALOIS_ALT:
      result = galoisFibAlt (N);
      break;

    case GALOIS_STACK:
      result = galoisFibStack (N);
      break;

    case GALOIS_GENERIC:
      result = galoisFibGeneric (N);
      break;


    default:
      std::abort ();

  }
  t.stop ();

  std::printf ("%dth Fibonacci number is: %d\n", unsigned(N), result);

  if (!skipVerify) {
    unsigned ser = serialFib (N);
    if (result != ser) {
      GALOIS_DIE("Result doesn't match with serial: ", ser);
    }
    else {
      std::printf ("OK... Result verifed ...\n");
    }
  }

  return 0;
}
