#include "Galois/Runtime/CacheManager.h"

using namespace Galois::Runtime;

struct foo {
  int x;
  int y;
};

int main() {
  auto& cm = getCacheManager();
  fatPointer fp{1, reinterpret_cast<void*>(0x010)};

  std::cout << fp << "\n";
  std::cout << cm.resolve(fp, false) << " " 
            << cm.resolve(fp, true) << "\n";

  cm.create(fp, false, foo{1,2});

  std::cout << fp << "\n";
  std::cout << cm.resolve(fp, false) << " " 
            << cm.resolve(fp, true) << "\n";

  cm.create(fp, true, foo{2,3});

  std::cout << fp << "\n";
  std::cout << cm.resolve(fp, false) << " " 
            << cm.resolve(fp, true) << "\n";


  return 0;
}
