include(CheckCXXSourceCompiles)
set(Cilk_CXX_TEST_SOURCE
"
#include <cilk/cilk.h>
int main(){ cilk_for(int i=0;i<1; ++i); }
")
CHECK_CXX_SOURCE_COMPILES("${Cilk_CXX_TEST_SOURCE}" HAVE_CILK)
if(HAVE_CILK)
  message(STATUS "A compiler with CILK support found")
endif()
