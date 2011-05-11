cmake_minimum_required (VERSION 2.6)
project (Galois)
site_name(SITE_NAME)

if (NOT CMAKE_BUILD_TYPE)
 		message(STATUS "No build type selected, default to Release")
 		set(CMAKE_BUILD_TYPE "Release")
endif()

###### Options ######
#use pthreads
set(PTHREAD ON)

#compile for cray xmt
#SET( CRAY ON )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

if(SITE_NAME MATCHES "volta.*")
  set(NUMA ON)
endif()

###### find libraries ######

#Boost
set(Boost_ADDITIONAL_VERSIONS "1.40" "1.40.0")
find_package( Boost 1.34.0 REQUIRED )
include_directories(${Boost_INCLUDE_DIRS})

#PThreads
if(PTHREAD)
  find_package(Threads REQUIRED)
  add_definitions(-DGALOIS_PTHREAD)
endif(PTHREAD)

if(NUMA)
  add_definitions(-DGALOIS_NUMA)
endif(NUMA)
#vtune
#apps/CMakeLists.txt also hardcodes this path
if(EXISTS "/opt/intel/vtune_amplifier_xe_2011/include/ittnotify.h")
  set(WITH_VTUNE ON)
endif()

if(CRAY)
  add_definitions(-DGALOIS_CRAY)
endif(CRAY)

#solaris
if(CMAKE_SYSTEM MATCHES "SunOs.*")
  add_definitions("-mpcu=niagra2")
  add_definitions("-lposix4")
endif()

###### Flags ######

#set(CMAKE_CXX_COMPILER /h1/lenharth/local/bin/clang++)
#set(CMAKE_CXX_COMPILER /opt/intel/Compiler/11.1/072/bin/intel64/icpc)
#These help the intel compiler
#add_definitions("-wd981 -wd383 -wd869 -wd2196")

#this is stupid, but imported libraries are scoped by directory
#add_library(hwloc SHARED IMPORTED)
#set_property(TARGET hwloc PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/lib/install/lib/libhwloc.so)
#include_directories(${CMAKE_BINARY_DIR}/lib/install/include)

set(CMAKE_C_FLAGS_DEBUG
  "-g -Wall -Wno-sign-compare" CACHE STRING
  "C flags for debug target" FORCE )
set(CMAKE_CXX_FLAGS_DEBUG
  "-g -Wall -Wno-sign-compare" CACHE STRING
  "CXX flags for debug target" FORCE )
set(CMAKE_C_FLAGS_RELWITHDEBINFO
  "-g -O3" CACHE STRING
  "CXX Flags for release with debug info target" FORCE )
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
  "-g -O3" CACHE STRING
  "CXX Flags for release with debug info target" FORCE )
set(CMAKE_C_FLAGS_RELEASE
  "-g -DNDEBUG -O3" CACHE STRING
  "C flags for release target" FORCE )
set(CMAKE_CXX_FLAGS_RELEASE
  "-g -DNDEBUG -O3" CACHE STRING
  "CXX flags for release target" FORCE )

###### Source finding ######
include_directories("include")

add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(tools)

###### Documentation ######
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(UseDoxygen OPTIONAL)
