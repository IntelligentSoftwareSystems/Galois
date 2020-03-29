# Find architecture-specific flags
#
# Once done this will define
#  ARCH_FLAGS_FOUND
#  ARCH_CXX_FLAGS - Compiler flags to enable architecture-specific optimizations
#  ARCH_C_FLAGS - Compiler flags to enable architecture-specific optimizations
#  ARCH_LINK_FLAGS - Compiler flags to enable architecture-specific optimizations
include(CheckCXXCompilerFlag)

if(NOT USE_ARCH OR USE_ARCH STREQUAL "none" OR ARCH_FLAGS_FOUND)
  set(ARCH_CXX_FLAGS_CANDIDATES)
else()
  set(ARCH_CXX_FLAGS_CANDIDATES "-march=${USE_ARCH}")
endif()

if(USE_ARCH STREQUAL "mic")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    list(APPEND ARCH_CXX_FLAGS_CANDIDATES -mmic)
  endif()

  if(CMAKE_COMPILER_IS_GNUCC)
    list(APPEND ARCH_CXX_FLAGS_CANDIDATES -march=knc)
  endif()
endif()

foreach(FLAG ${ARCH_CXX_FLAGS_CANDIDATES})
  message(STATUS "Try architecture flag = [${FLAG}]")
  unset(ARCH_CXX_FLAGS_DETECTED)
  check_cxx_compiler_flag("${FLAG}" ARCH_CXX_FLAGS_DETECTED)
  if(ARCH_CXX_FLAGS_DETECTED)
    set(ARCH_FLAGS_FOUND "YES")
    set(ARCH_CXX_FLAGS "${FLAG}")
    set(ARCH_C_FLAGS "${FLAG}")
    set(ARCH_LINK_FLAGS "${FLAG}")
  endif()
endforeach()
