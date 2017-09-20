#include "galois/Galois.h"
#include "galois/Runtime/Directory.h"

using namespace std;
using namespace galois::Runtime;

struct simple :public Lockable {
  int member;
};

int main(int argc, char *argv[])
{

  DirectoryNG dir;

  simple foo;
  gptr<simple> bar(1, &foo);
  fatPointer baz(1, &foo);

  dir.dump(std::cout, baz);
  std::cout << "\n";

  std::cout << "resolve RO: ";
  dir.resolve<simple>(baz, RO);
  dir.dump(std::cout, baz);
  std::cout << "\n";

  std::cout << "resolve RO: ";
  dir.resolve<simple>(baz, RO);
  dir.dump(std::cout, baz);
  std::cout << "\n";

  std::cout << "recvObj RO: ";
  dir.recvObj(baz, &foo, RO);
  dir.dump(std::cout, baz);
  std::cout << "\n";

  std::cout << "resolve RW: ";
  dir.resolve<simple>(baz, RW);
  dir.dump(std::cout, baz);
  std::cout << "\n";

  std::cout << "recvObj INV: ";
  dir.recvObj(baz, nullptr, INV);
  dir.dump(std::cout, baz);
  std::cout << "\n";

  std::cout << "recvObj RW: ";
  dir.recvObj(baz, &foo, RW);
  dir.dump(std::cout, baz);
  std::cout << "\n";

  std::cout << "recvObj INV: ";
  dir.recvObj(baz, nullptr, INV);
  dir.dump(std::cout, baz);
  std::cout << "\n";

  return 0;
}
