# Find architecture-specific flags
# Once done this will define
#  ARCH_CXX_FLAGS - Compiler flags to enable architecture-specific optimizations
#  ARCH_C_FLAGS - Compiler flags to enable architecture-specific optimizations
#  ARCH_EXE_FLAGS - Compiler flags to enable architecture-specific optimizations
include(CheckCXXCompilerFlag)

set(ARCH_CXX_FLAGS_CANDIDATES "-march=${USE_ARCH}")

if(ARCH STREQUAL "mic")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    list(APPEND ARCH_CXX_FLAGS_CANDIDATES -mmic)
  endif()

  if(CMAKE_COMPILER_IS_GNUCC)
    list(APPEND ARCH_CXX_FLAGS_CANDIDATES -march=knc)
  endif()
endif()

# Don't do anything when already set
if(ARCH_CXX_FLAGS)
  set(ARCH_CXX_FLAGS_CANDIDATES)
  set(ARCH_FLAGS_FOUND_INTERNAL "YES")
endif()

foreach(FLAG ${ARCH_CXX_FLAGS_CANDIDATES})
  unset(ARCH_CXX_FLAGS_DETECTED CACHE)
  message(STATUS "Try C++ architecture flag = [${FLAG}]")
  check_cxx_compiler_flag("${FLAG}" ARCH_CXX_FLAGS_DETECTED)
  if(ARCH_CXX_FLAGS_DETECTED)
    set(ARCH_FLAGS_FOUND_INTERNAL "YES")
    set(ARCH_CXX_FLAGS "${FLAG}" CACHE STRING "C++ compiler flags for architecture-specific optimizations")
    set(ARCH_C_FLAGS "${FLAG}" CACHE STRING "C compiler flags for architecture-specific optimizations")
    set(ARCH_EXE_FLAGS "${FLAG}" CACHE STRING "Linker flags for architecture-specific optimizatinos")
    break()
  endif()
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ARCH_FLAGS DEFAULT_MSG ARCH_FLAGS_FOUND_INTERNAL)
mark_as_advanced(ARCH_CXX_FLAGS)
mark_as_advanced(ARCH_C_FLAGS)
mark_as_advanced(ARCH_EXE_FLAGS)
