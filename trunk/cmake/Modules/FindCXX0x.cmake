include(CheckCXXCompilerFlag)

set(CXX0X_FLAG_CANDIDATES -std=c++0x -std=c++11x)

# Don't do anything when already set
if(DEFINED CXX0X_FLAGS)
  set(CXX0X_FLAG_CANDIDATES)
  set(CXX0X_FOUND_INTERNAL "YES")
endif()

foreach(FLAG ${CXX0X_FLAG_CANDIDATES})
  unset(CXX0X_FLAG_DETECTED CACHE)
  message(STATUS "Try C++0x flag = [${FLAG}]")
  check_cxx_compiler_flag("${FLAG}" CXX0X_FLAG_DETECTED)
  if(CXX0X_FLAG_DETECTED)
    set(CXX0X_FOUND_INTERNAL "YES")
    set(CXX0X_FLAGS "${FLAG}" CACHE STRING "C++ compiler flags for C++0x features")
    break()
  endif()
endforeach()

find_package_handle_standard_args(CXX0x DEFAULT_MSG CXX0X_FOUND_INTERNAL CXX0X_FLAGS)
mark_as_advanced(CXX0X_FLAGS)

if(CXX0X_FLAG_DETECTED)
  set(_oldflags ${CMAKE_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS "${_oldflags} ${CXX0X_FLAGS}")
  include(CheckCXX0xFeatures)
  set(CMAKE_CXX_FLAGS ${_oldflags})
else()
  include(CheckCXX0xFeatures)
endif()
