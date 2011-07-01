cmake_minimum_required (VERSION 2.6)
project (Galois)
site_name(SITE_NAME)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

###### Options (alternatively pass as options to cmake -DName=Value ######

#use pthreads
set(USE_PTHREAD ON)

#compile for cray xmt
#set(USE_CRAY ON)

#use icc
#set(USE_ICC ON)

#use profiling
#set(USE_PROFILING ON)

##### Machine-specific settings (probably not useful to you) ######
if(SITE_NAME MATCHES "volta.*")
  set(USE_NUMA ON)
endif()

if(SITE_NAME MATCHES "delaunay.*")
  set(CMAKE_CXX_COMPILER /opt/csw/gcc4/bin/g++)
  set(CMAKE_CC_COMPILER /opt/csw/gcc4/bin/gcc)
  # Override broken include files in
  #  /opt/csw/gcc4/lib/gcc/sparc-sun-solaris2.8/4.5.1/include-fixed/
  add_definitions(-nostdinc)
  add_definitions(-I/usr/include)
  add_definitions(-I/opt/csw/gcc4/include/c++/4.5.1)
  add_definitions(-I/opt/csw/gcc4/include/c++/4.5.1/sparc-sun-solaris2.8)
endif()

if(CMAKE_SYSTEM MATCHES "SunOS.*")
  set(USE_SOLARIS ON)
endif()

###### Configure (users don't need to go beyond here) ######

#Boost
set(Boost_ADDITIONAL_VERSIONS "1.40" "1.40.0")
find_package( Boost 1.34.0 REQUIRED )
include_directories(${Boost_INCLUDE_DIRS})

#PThreads
if(USE_PTHREAD)
  #find_package(Threads REQUIRED)
  find_package(Threads)
  add_definitions(-DGALOIS_PTHREAD)
endif(USE_PTHREAD)

if(USE_NUMA)
  add_definitions(-DGALOIS_NUMA)
endif(USE_NUMA)

#vtune
set(VTUNE_INCLUDE "/opt/intel/vtune_amplifier_xe_2011/include")
set(VTUNE_LIB "/opt/intel/vtune_amplifier_xe_2011/lib64/libittnotify.a")
if(EXISTS ${VTUNE_INCLUDE})
  set(USE_VTUNE ON)
endif()

if(USE_CRAY)
  add_definitions(-DGALOIS_CRAY)
endif(USE_CRAY)

if(USE_PROFILING)
  add_definitions(-pg)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif(USE_PROFILING)

#solaris
if(USE_SOLARIS)
  add_definitions(-DSOLARIS)
  add_definitions("-m64")
  add_definitions("-mcpu=niagara2")
  add_definitions("-lposix4")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -m64 -lposix4")
endif()

if(USE_ICC)
  set(CMAKE_CXX_COMPILER /opt/intel/Compiler/11.1/072/bin/intel64/icpc)
  #These help the intel compiler
  add_definitions("-wd981 -wd383 -wd869 -wd2196 -wd279")
endif(USE_ICC)

#set(CMAKE_CXX_COMPILER /h1/lenharth/local/bin/clang++)

#Always include debug symbols
add_definitions("-g")

#Disable asserts
if(CMAKE_BUILD_TYPE MATCHES "Release")
  add_definitions("-DNDEBUG")
endif()

#more warnings
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  add_definitions("-Wall")
endif()

###### Source finding ######
include_directories("include")

add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(tools)
add_subdirectory(inputs)

###### Documentation ######
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(UseDoxygen OPTIONAL)
