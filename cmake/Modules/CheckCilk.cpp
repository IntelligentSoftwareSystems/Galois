#include <cilk/cilk.h>

int main(int c, char** argv) {
  cilk_for (int i=0; i<4; ++i) {
    ;
  }
  return 0;
}
