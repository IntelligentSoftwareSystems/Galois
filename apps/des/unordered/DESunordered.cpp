#include "DESunordered.h"

int main (int argc, const char* argv[]) {

  DESunordered* m = new DESunordered();
  m->run (argc, argv);
  delete m;

  return 0;
}
