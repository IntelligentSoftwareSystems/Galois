#include <iostream>

#include "Support/ThreadSafe/cray_simple_lock.h"

int main() {

  threadsafe::cray::simpleLock L;

  L.write_lock();
  std::cout << "Locked\n";
  L.write_unlock();
  std::cout << "Unlocked\n";
  L.write_lock();
  std::cout << "Locked(2)\n";
  L.write_unlock();
  std::cout << "Unlocked(2)\n";

  std::cout << "Done\n";
  return 0;
}
