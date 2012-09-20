#include "Galois/Timer.h"
#include "Galois/Runtime/Context.h"

#include <iostream>

int main(int argc, char** argv) {

  GaloisRuntime::SimpleRuntimeContext S;
  GaloisRuntime::Lockable L;


  Galois::Timer t;
  t.start();
  
  for (int x = 0; x < 1024*1024*1024; ++x)
    GaloisRuntime::doAcquire(&L);
  
  t.stop();
  std::cout << "Locking: " << t.get() << '\n';
  
  return 0;
}
