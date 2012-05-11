#include "Exp/Parallel.h"

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
