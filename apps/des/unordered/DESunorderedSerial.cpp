#include "DESunorderedSerial.h"

int main (int argc, const char* argv[]) {

  DESunorderedSerial* s = new DESunorderedSerial();
  s->run (argc, argv);
  delete s;

  return 0;
}
