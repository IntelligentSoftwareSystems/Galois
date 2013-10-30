# Find C++11 flags
# Once done this will define
#  CXX11_FLAGS - Compiler flags to enable C++11
include(CheckCXXCompilerFlag)

# This covers gcc, icc, clang, xlc

# Place xlc (-qlanglvl=extended0x) first because xlc parses -std but does not
# halt even with -qhalt=i
set(CXX11_FLAG_CANDIDATES -qlanglvl=extended0x -std=c++11 -std=c++0x)

# some versions of cmake don't recognize clang's rejection of unknown flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CXX11_FLAG_CANDIDATES -std=c++11 -std=c++0x)
endif()

# Don't do anything when already set
if(CXX11_FLAGS)
  set(CXX11_FLAG_CANDIDATES)
  set(CXX11_FOUND_INTERNAL "YES")
endif()

foreach(FLAG ${CXX11_FLAG_CANDIDATES})
  unset(CXX11_FLAG_DETECTED CACHE)
  message(STATUS "Try C++11 flag = [${FLAG}]")
  check_cxx_compiler_flag("${FLAG}" CXX11_FLAG_DETECTED)
  if(CXX11_FLAG_DETECTED)
    set(CXX11_FOUND_INTERNAL "YES")
    set(CXX11_FLAGS "${FLAG}" CACHE STRING "C++ compiler flags for C++11 features")
    break()
  endif()
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CXX11 DEFAULT_MSG CXX11_FOUND_INTERNAL CXX11_FLAGS)
mark_as_advanced(CXX11_FLAGS)

include(CheckCXX11Features)
