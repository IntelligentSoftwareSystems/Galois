

template<typename T>
struct checker {
  T wl;
  typename T::template rethread<true>::WL wl2;
  typename T::template rethread<true>::WL wl3;

  checker() {
    int a[2] = {1,2};
    wl.fill_initial(&a[0], &a[2]);
    wl.push(0);
    wl.pop();
    wl.empty();
    wl.aborted(34);

    wl2.fill_initial(&a[0], &a[2]);
    wl2.push(0);
    wl2.pop();
    wl2.empty();
    wl2.aborted(34);

    wl3.fill_initial(&a[0], &a[2]);
    wl3.push(0);
    wl3.pop();
    wl3.empty();
    wl3.aborted(34);

  }
};


#define WLCOMPILECHECK(name) checker<name<int> > ck_##name;
#include "Galois/Runtime/WorkList.h"

int main() {
  return 0;
}
