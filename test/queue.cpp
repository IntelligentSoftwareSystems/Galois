#include "Galois/Queue.h"
#include <iostream>
void check(const std::pair<bool,int>& r, int exp) {
  if (r.first && r.second == exp)
    ;
  else {
    std::cerr << "Expected " << exp << "\n";
    abort();
  }
}

int main() {
  Galois::ConcurrentSkipListMap<int,int> map;
  int v = 0;

  for (int i = 100; i >= 0; --i) {
    map.put(i, &v);
  }
  for (int i = 0; i <= 100; ++i) {
    check(map.pollFirstKey(), i);
  }
  return 0;
}
