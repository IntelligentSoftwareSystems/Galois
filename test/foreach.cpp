#include "galois/Galois.h"
#include "galois/Bag.h"
#include <vector>
#include <iostream>

void function_pointer(int x, galois::UserContext<int>& ctx) {
  std::cout << x << "\n";  
}

struct function_object {
  void operator()(int x, galois::UserContext<int>& ctx) const {
    function_pointer(x, ctx);
  }
};

int main() {
  std::vector<int> v(10);
  galois::InsertBag<int> b;

  galois::for_each(v.begin(), v.end(), &function_pointer);
  galois::for_each(v.begin(), v.end(), function_object(), galois::loopname("with function object and options"));
  galois::do_all(v.begin(), v.end(), [&b](int x) { b.push(x); });
  galois::for_each_local(b, function_object());
  
  // Works without context as well
#if defined(__INTEL_COMPILER) && __INTEL_COMPILER <= 1400
#else
  //Don't support Context-free versions yet (gcc 4.7 problem)
  //  galois::for_each(v.begin(), v.end(), [](int x) { std::cout << x << "\n"; });
  //galois::for_each_local(b, [](int x) { std::cout << x << "\n"; });
#endif

  return 0;
}
