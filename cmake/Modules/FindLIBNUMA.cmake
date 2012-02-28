include(CheckCSourceCompiles)

set (LIBNUMA_FOUND "NO")

find_library(LIBNUMA numa)
if(LIBNUMA_FOUND)
  CHECK_C_SOURCE_COMPILES("${CMAKE_MODULE_PATH}/FindLIBNUMA.c" LIBNUMA_FOUND)
endif()
