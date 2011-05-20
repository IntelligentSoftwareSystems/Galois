/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

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
