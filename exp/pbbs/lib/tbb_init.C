
#ifdef EXP_DOALL_TBB

#include "tbb/task_scheduler_init.h"
#include <cstdlib>

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

Init iii;

#endif

void tbb_init() {
  // external reference to cause the initialization of static object above
}

