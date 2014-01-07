#include "Galois/Runtime/Lockable.h"

#include <iostream>

using namespace Galois::Runtime;

struct simple : public Lockable {
  int foo;
};

char translate(int i) {
  switch (i) {
  case 0:
    return 'F';
  case 1:
    return 'N';
  case 3:
    return 'O';
  default:
    return '?';
  }
}

int main(int argc, char** argv) {
  simple s1, s2;
  LockManagerBase b1, b2;

  std::cout << translate(b1.tryAcquire(&s1)) << "\n";
  b1.dump(std::cout); b2.dump(std::cout); std::cout << "\n";
  std::cout << translate(b1.tryAcquire(&s1)) << "\n";
  b1.dump(std::cout); b2.dump(std::cout); std::cout << "\n";
  std::cout << translate(b1.tryAcquire(&s2)) << "\n";
  b1.dump(std::cout); b2.dump(std::cout); std::cout << "\n";
  std::cout << translate(b2.tryAcquire(&s1)) << "\n";
  b1.dump(std::cout); b2.dump(std::cout); std::cout << "\n";
  std::cout << translate(b2.tryAcquire(&s2)) << "\n";
  b1.dump(std::cout); b2.dump(std::cout); std::cout << "\n";
  std::cout << b1.releaseAll() << "\n";
  b1.dump(std::cout); b2.dump(std::cout); std::cout << "\n";
  std::cout << translate(b2.tryAcquire(&s1)) << "\n";
  b1.dump(std::cout); b2.dump(std::cout); std::cout << "\n";
  std::cout << translate(b2.tryAcquire(&s2)) << "\n";
  b1.dump(std::cout); b2.dump(std::cout); std::cout << "\n";
  b1.forceAcquire(&s1);
  b1.dump(std::cout); b2.dump(std::cout); std::cout << "\n";
  b1.forceAcquire(&s2);
  b1.dump(std::cout); b2.dump(std::cout); std::cout << "\n";
  std::cout << translate(b2.tryAcquire(&s1)) << "\n";
  b1.dump(std::cout); b2.dump(std::cout); std::cout << "\n";
  std::cout << translate(b2.tryAcquire(&s2)) << "\n";
  b1.dump(std::cout); b2.dump(std::cout); std::cout << "\n";
  std::cout << b2.releaseAllChecked() << "\n";
  b1.dump(std::cout); b2.dump(std::cout); std::cout << "\n";
  std::cout << b1.releaseAllChecked() << "\n";
  b1.dump(std::cout); b2.dump(std::cout); std::cout << "\n";

  return 0;
}
