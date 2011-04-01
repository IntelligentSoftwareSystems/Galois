cmake_minimum_required (VERSION 2.6)
project (Galois)

#Options

#use pthreads
SET( PTHREAD ON)

#compile for cray xmt
#SET( CRAY ON )

#compile with debug symbols (recommended)
SET( WITH_DEBUG_INFO ON )

#be extra pendantic (recommended)
#SET( WITH_WALL ON)

#disable expensive asserts (use on Release builds)
SET( WITH_NO_ASSERTS ON )

#build with optimizations (release build)
SET( WITH_OPTIMIZATIONS ON)

#find libraries

#Boost
SET(Boost_ADDITIONAL_VERSIONS "1.40" "1.40.0")
find_package( Boost 1.34.0 REQUIRED )
include_directories(${Boost_INCLUDE_DIRS})

#PThreads
if (PTHREAD)
  find_package(Threads REQUIRED)
  add_definitions(-DGALOIS_PTHREAD)
endif(PTHREAD)

#vtune
#apps/CMakeLists.txt also hardcodes this path
IF(EXISTS "/opt/intel/vtune_amplifier_xe_2011/include/ittnotify.h")
  SET(WITH_VTUNE ON)
endif()

if (CRAY)
  add_definitions(-DGALOIS_CRAY)
endif(CRAY)

if (WITH_DEBUG_INFO)
  add_definitions("-g")
endif(WITH_DEBUG_INFO)

if (WITH_WALL)
  add_definitions("-Wall -Wno-sign-compare")
endif(WITH_WALL)

if(WITH_NO_ASSERTS)
  add_definitions("-DNDEBUG")
endif(WITH_NO_ASSERTS)

#solaris
IF(CMAKE_SYSTEM MATCHES "SunOs.*")
  add_definitions("-mpcu=niagra2")
  add_definitions("-lposix4")
endif()

#Flags

set(CMAKE_CXX_FLAGS_RELEASE         "-O3")

if (WITH_OPTIMIZATIONS)
  set(CMAKE_BUILD_TYPE Release)
else(WITH_OPTIMIZATIONS)
  set(CMAKE_BUILD_TYPE Debug)
endif(WITH_OPTIMIZATIONS)

#set(CMAKE_CXX_COMPILER /h1/lenharth/local/bin/clang++)
#set(CMAKE_CXX_COMPILER /opt/intel/Compiler/11.1/072/bin/intel64/icpc)
#These help the intel compiler
#add_definitions("-wd981 -wd383 -wd869 -wd2196")

#this is stupid, but imported libraries are scoped by directory
#add_library(hwloc SHARED IMPORTED)
#set_property(TARGET hwloc PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/lib/install/lib/libhwloc.so)
#include_directories(${CMAKE_BINARY_DIR}/lib/install/include)

#Source finding
include_directories("include")

add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(tools)

#Documentation
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(UseDoxygen OPTIONAL)
