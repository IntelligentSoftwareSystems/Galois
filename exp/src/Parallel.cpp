#include "Exp/Parallel.h"
#include "Galois/Galois.h"

__thread unsigned Exp::TID = 0;
unsigned Exp::nextID = 0;

unsigned Exp::getNumThreads() {
  char *p = getenv("GALOIS_NUM_THREADS");
  if (p) {
    int n = atoi(p);
    if (n > 0)
      return n;
  }
  return 1;
}

int Exp::getNumRounds() {
  char *p = getenv("EXP_NUM_ROUNDS");
  if (p) {
    int n = atoi(p);
    if (n > 0)
      return n;
  }
  return -1;
}

#ifdef EXP_DOALL_GALOIS
struct Init {
  Init() {
    Galois::setMaxThreads(Exp::getNumThreads()); 
  }
};
#endif

#ifdef EXP_DOALL_TBB
#include "tbb/task_scheduler_init.h"
struct Init {
  tbb::task_scheduler_init* init;

  int get_threads() {
    char *p = getenv("TBB_NUM_THREADS");
    if (p) {
      int n = atoi(p);
      if (n > 0)
        return n;
    }
    return 1;
  }
  
  Init() {
    int n = get_threads();
    init = new tbb::task_scheduler_init(n);  
  }

  ~Init() {
    if (init)
      delete init;
  }
};
#endif

#if defined(EXP_DOALL_CILK) || defined(EXP_DOALL_CILKP) || defined(EXP_DOALL_OPENMP) || defined(EXP_DOALL_OPENMP_RUNTIME)
struct Init { };
#endif

namespace {
Init iii;
} // end namespace

void Exp::initRuntime() {
  // external reference to cause the initialization of static object above
}

