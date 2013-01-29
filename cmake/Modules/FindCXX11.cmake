include(CheckCXXCompilerFlag)

#this covers gcc, icc, clang
set(CXX11_FLAG_CANDIDATES -std=c++11 -std=c++0x)

# Don't do anything when already set
if(DEFINED CXX11_FLAGS)
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

find_package_handle_standard_args(CXX11 DEFAULT_MSG CXX11_FOUND_INTERNAL CXX11_FLAGS)
mark_as_advanced(CXX11_FLAGS)

include(CheckCXX11Features)

