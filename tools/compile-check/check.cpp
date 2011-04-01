

template<typename T>
struct checker {
  T wl;
  void foo() {
    wl.push(0);
    wl.pop();
    wl.empty();
    wl.aborted(34);
  }
};


#define WLCOMPILECHECK(name) checker<name<int> > ck_##name;
#include "Galois/Runtime/WorkList.h"

int main() {
  return 0;
}
