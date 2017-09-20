#include <iostream>

#include <tbb/tbb.h>


#include "Galois/Galois.h"
#include "Galois/CilkInit.h"
#include "Galois/Timer.h"
#include "Galois/Runtime/TreeExec.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

namespace cll = llvm::cl;
static cll::opt<unsigned> N("n", cll::desc("n-th fibonacci number"), cll::init(39));

enum ExecType {
  SERIAL, CILK, GALOIS, GALOIS_ALT, GALOIS_STACK, GALOIS_GENERIC, HAND, OPENMP, TBB
};

static cll::opt<ExecType> execType (
    cll::desc ("executor type"),
    cll::values (
      clEnumVal (SERIAL, "serial recursive"),
      clEnumVal (CILK, "CILK divide and conquer implementation"),
      clEnumVal (GALOIS, "galois basic divide and conquer implementation"),
      clEnumVal (GALOIS_STACK, "galois using thread stack"),
      clEnumVal (GALOIS_GENERIC, "galois std::function version"),
      clEnumVal (HAND, "Andrew's Handwritten version"),
      clEnumVal (OPENMP, "OpenMP implementation"),
      clEnumVal (TBB, "TBB implementation"),
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


struct FibRecord {
  unsigned n;
  unsigned* result;
  unsigned term_n_1;
  unsigned term_n_2;
};

struct GaloisDivide {
  template <typename C>
  void operator () (FibRecord& r, C& ctx) {
    if (r.n <= 2) {
      r.term_n_1 = r.n;
      r.term_n_2 = 0;
      return;
    }

    FibRecord left {r.n-1, &(r.term_n_1), 0, 0 };

    FibRecord rigt {r.n-2, &(r.term_n_2), 0, 0 };

    ctx.spawn (left);
    ctx.spawn (rigt);
  }
};

struct GaloisConquer {
  void operator () (FibRecord& r) {
    *(r.result) = r.term_n_1 + r.term_n_2;
  };
};

unsigned galoisFib (unsigned n) {

  unsigned result = 0;

  FibRecord init { n, &result, 0, 0};

  galois::runtime::for_each_ordered_tree (
      init,
      GaloisDivide (),
      GaloisConquer (),
      "fib-galois");

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

  galois::runtime::for_each_ordered_tree (init, "fib");

  return init.result;
}

struct GaloisFibGeneric: public galois::runtime::TreeTaskBase {
  unsigned n;
  unsigned result;

  GaloisFibGeneric (unsigned _n, unsigned _result): 
    galois::runtime::TreeTaskBase (),
    n (_n),
    result (_result)
  {}

  virtual void operator () (galois::runtime::TreeTaskContext& ctx) {
    if (n <= 2) {
      result = n;
      return;
    }

    GaloisFibGeneric left {n-1, 0};
    ctx.spawn (left);

    GaloisFibGeneric right {n-2, 0};
    ctx.spawn (right);

    ctx.sync ();

    result = left.result + right.result;
  }
};

unsigned galoisFibGeneric (unsigned n) {
  GaloisFibGeneric init {n, 0};

  galois::runtime::for_each_ordered_tree_generic (init, "fib-gen");
  return init.result;
}


struct FibHandFrame {
  std::atomic<int> sum;
  std::atomic<int> done;
  FibHandFrame* parent;
};

galois::InsertBag<FibHandFrame> B;

struct FibHandOp {
  typedef int tt_does_not_need_aborts;
  typedef int tt_does_not_need_stats;

  void notify_parent(FibHandFrame* r, int val) {
    if (!r) return;
    //fastpath
    if (r->done == 1) {
      notify_parent(r->parent, val + r->sum);
      return;
    }
    r->sum += val;
    if (++r->done == 2) {
      notify_parent(r->parent, r->sum);
      return;
    } //else, someone else will clean up
  }

  template<typename ContextTy>
  void operator() (std::pair<int, FibHandFrame*> wi, ContextTy& ctx) {
    int n = wi.first;
    FibHandFrame* r = wi.second;
    if (n <= 2) {
      notify_parent(r, n);
      return;
    }
    FibHandFrame& foo = B.emplace();
    foo.sum = 0;
    foo.done = 0;
    foo.parent = r;
    ctx.push(std::make_pair(n-1, &foo));
    ctx.push(std::make_pair(n-2, &foo));
    return;
  }
};

unsigned fibHand (unsigned n) {

  typedef galois::worklists::AltChunkedFIFO<64> Chunked;
  // typedef galois::worklists::AltChunkedLIFO<4> Chunked;

  FibHandFrame init;
  init.sum = 0;
  init.done = 0;
  init.parent = 0;

  galois::for_each(std::make_pair(n, &init), 
      FibHandOp(), 
      galois::loopname ("fib-hand"),
      galois::wl<Chunked>());

  return init.sum;
}

unsigned fibOMP (unsigned n) {
  if (n <= 2)
    return n;
  unsigned x, y;
#pragma omp task shared(x) firstprivate(n) untied
  x = fibOMP (n-1);
#pragma omp task shared(y) firstprivate(n) untied
  y = fibOMP (n-2);
#pragma omp taskwait
  return x + y;
}

unsigned fibOpenMP (unsigned n) {
  unsigned result = 0;
// #pragma omp parallel shared(result, n)
#pragma omp parallel 
  {
#pragma omp single nowait 
    result = fibOMP(n);
  }

  return result;
}

// struct FibTBBFrame {
  // unsigned n;
  // unsigned* result;
// 
  // void operator () (void) {
    // if (n <= 2) { 
      // *result = n;
      // return;
    // }
// 
    // unsigned x;
    // FibTBBFrame left {n-1, &x};
// 
    // unsigned y;
    // FibTBBFrame right {n-2, &y};
// 
    // tbb::parallel_invoke (left, right);
// 
    // *result = x + y;
  // }
// };

class FibTBBTask: public tbb::task {
public:
  const unsigned n;
  unsigned* const sum;

  FibTBBTask( unsigned n_, unsigned* sum_ ) :
    n(n_), sum(sum_)
  {}
  tbb::task* execute() {      // Overrides virtual function task::execute
    if( n <= 2 ) {
      *sum = n;
    } else {
      unsigned x, y;
      FibTBBTask& a = *new( allocate_child() ) FibTBBTask(n-1,&x);
      FibTBBTask& b = *new( allocate_child() ) FibTBBTask(n-2,&y);
      set_ref_count(3);
      // Start b running.
      spawn( b );
      // Start a running and wait for all children (a and b).
      spawn_and_wait_for_all(a);
      // Do the sum
      *sum = x+y;
    }
    return NULL;
  }
};
                    


unsigned fibTBB (unsigned n) {
  tbb::task_scheduler_init init;

  unsigned result = 0;
  FibTBBTask root {n, &result};
  (&root)->execute ();
  return result;
};

int main (int argc, char* argv[]) {

  galois::StatManager sm;
  LonestarStart (argc, argv, name, desc, url);

  unsigned result = -1;

  galois::StatTimer t;

  t.start ();
  switch (execType) {
    case SERIAL:
      result = serialFib (N);
      break;

    case CILK:
      galois::CilkInit ();
      result = fib (N);
      break;

    case GALOIS:
      result = galoisFib (N);
      break;

    case GALOIS_STACK:
      result = galoisFibStack (N);
      break;

    case GALOIS_GENERIC:
      result = galoisFibGeneric (N);
      break;

    case HAND:
      result = fibHand (N);
      break;

    case OPENMP:
      result = fibOpenMP (N);
      break;

    case TBB:
      result = fibTBB (N);
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
