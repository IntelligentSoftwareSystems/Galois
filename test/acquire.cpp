#include "Galois/Timer.h"
#include "Galois/Runtime/Context.h"

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
  galois::Runtime::SimpleRuntimeContext S;
  galois::Runtime::Lockable L;

  int numAcquires = 1;
  if (argc > 1)
    numAcquires = atoi(argv[1]);
  if (numAcquires <= 0)
    numAcquires = 1024*1024*1024;

  galois::Timer t;
  t.start();
 
  for (int x = 0; x < numAcquires; ++x)
    galois::Runtime::acquire(&L, galois::MethodFlag::WRITE);
  
  t.stop();
  std::cout << "Locking time: " << t.get() << " ms after " << numAcquires << "\n";
  
  return 0;
}
