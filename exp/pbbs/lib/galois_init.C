
#ifdef EXP_DOALL_GALOIS

#include "Galois/Galois.h"

#include <cstdlib>

struct Init {
  int get_threads() {
    char *p = getenv("GALOIS_NUM_THREADS");
    if (p) {
      int n = atoi(p);
      if (n > 0)
        return n;
    }
    return 1;
  }
  
  Init() {
    int n = get_threads();
    Galois::setMaxThreads(n); 
  }

};

Init iii;

#endif
void galois_init() {
  // external reference to cause the initialization of static object above
}


