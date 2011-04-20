

template<typename T>
struct checker {
  T wl;
  checker() {
    int a[2] = {1,2};
    wl.fill_initial(&a[0], &a[2]);
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
