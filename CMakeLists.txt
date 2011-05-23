cmake_minimum_required (VERSION 2.6)
project (Galois)
site_name(SITE_NAME)

if (NOT CMAKE_BUILD_TYPE)
 		message(STATUS "No build type selected, default to Release")
 		set(CMAKE_BUILD_TYPE "Release")
endif()

###### Options ######
#use pthreads
set(USE_PTHREAD ON)

#compile for cray xmt
#SET(USE_CRAY ON )

#use icc
#SET(USE_ICC ON)

if(SITE_NAME MATCHES "volta.*")
  set(USE_NUMA ON)
endif()

###### find libraries ######

#Boost
set(Boost_ADDITIONAL_VERSIONS "1.40" "1.40.0")
find_package( Boost 1.34.0 REQUIRED )
include_directories(${Boost_INCLUDE_DIRS})

#PThreads
if(USE_PTHREAD)
  find_package(Threads REQUIRED)
  add_definitions(-DGALOIS_PTHREAD)
endif(USE_PTHREAD)

if(USE_NUMA)
  add_definitions(-DGALOIS_NUMA)
endif(USE_NUMA)

#vtune
#apps/CMakeLists.txt also hardcodes this path
if(EXISTS "/opt/intel/vtune_amplifier_xe_2011/include/ittnotify.h")
  set(USE_VTUNE ON)
endif()

if(USE_CRAY)
  add_definitions(-DGALOIS_CRAY)
endif(USE_CRAY)

#solaris
if(CMAKE_SYSTEM MATCHES "SunOS.*")
  add_definitions("-m64")
  add_definitions("-mcpu=niagara2")
  add_definitions("-lposix4")
endif()

if(USE_ICC)
  set(CMAKE_CXX_COMPILER /opt/intel/Compiler/11.1/072/bin/intel64/icpc)
  #These help the intel compiler
  add_definitions("-wd981 -wd383 -wd869 -wd2196 -wd279")
endif(USE_ICC)

###### Flags ######

#set(CMAKE_CXX_COMPILER /h1/lenharth/local/bin/clang++)

#this is stupid, but imported libraries are scoped by directory
#add_library(hwloc SHARED IMPORTED)
#set_property(TARGET hwloc PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/lib/install/lib/libhwloc.so)
#include_directories(${CMAKE_BINARY_DIR}/lib/install/include)

#Always include debug symbols
add_definitions("-g")

#Disable asserts
if (CMAKE_BUILD_TYPE MATCHES "Release")
  add_definitions("-DNDEBUG")
endif()

#more warnings
if (CMAKE_BUILD_TYPE MATCHES "Debug")
  add_definitions("-Wall")
endif()

###### Source finding ######
include_directories("include")

add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(tools)

###### Documentation ######
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(UseDoxygen OPTIONAL)
