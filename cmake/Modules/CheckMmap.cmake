include(CheckCSourceCompiles)
set(Mmap64_C_TEST_SOURCE
"
#ifdef __linux__
#include <linux/mman.h>
#endif
#include <sys/mman.h>

int main(int c, char** argv) {
  void *ptr = mmap64(0, 2*1024*1024, PROT_READ|PROT_WRITE, MAP_PRIVATE, -1, 0);
  return 0;
}
")

if(HAVE_MMAP64)

else()
  CHECK_C_SOURCE_COMPILES("${Mmap64_C_TEST_SOURCE}" HAVE_MMAP64_INTERNAL)
  if(HAVE_MMAP64_INTERNAL)
    message(STATUS "mmap64 found")
    set(HAVE_MMAP64 "${HAVE_MMAP64_INTERNAL}" CACHE BOOL "Have mmap64")
  endif()
endif()
