#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include <vector>
#include <iostream>

void function_pointer(int x, Galois::UserContext<int>& ctx) {
  std::cout << x << "\n";  
}

struct function_object {
  void operator()(int x, Galois::UserContext<int>& ctx) const {
    function_pointer(x, ctx);
  }
};

int main() {
  std::vector<int> v(10);
  Galois::InsertBag<int> b;

  Galois::for_each(v.begin(), v.end(), &function_pointer);
  Galois::for_each(v.begin(), v.end(), function_object(), Galois::loopname("with function object and options"));
  Galois::do_all(v.begin(), v.end(), [&b](int x) { b.push(x); });
  Galois::for_each_local(b, function_object());
  
  // Works without context as well
  Galois::for_each(v.begin(), v.end(), [](int x) { std::cout << x << "\n"; });
  Galois::for_each_local(b, [](int x) { std::cout << x << "\n"; });

  return 0;
}
