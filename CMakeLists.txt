cmake_minimum_required (VERSION 2.6)
project (Galois)

#Options
SET( PTHREAD ON)
#SET( CRAY ON )
SET( WITH_VTUNE ON )

#Boost
SET(Boost_ADDITIONAL_VERSIONS "1.40" "1.40.0")
find_package( Boost 1.34.0 REQUIRED )
include_directories(${Boost_INCLUDE_DIRS})

#PThreads
if (PTHREAD)
  find_package(Threads REQUIRED)
  add_definitions(-DGALOIS_PTHREAD)
endif(PTHREAD)

if (CRAY)
  add_definitions(-DGALOIS_CRAY)
endif(CRAY)

#Flags
add_definitions("-Wall")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -g -DNDEBUG")
#set(CMAKE_CXX_FLAGS_RELEASE        "-DNDEBUG -pg")

set(CMAKE_CXX_FLAGS_DEBUG          "-g")

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

#set(CMAKE_CXX_COMPILER clang++)
#set(CMAKE_CXX_COMPILER /opt/intel/Compiler/11.1/073/bin/intel64/icpc)
#These help the intel compiler
#add_definitions("-wd981")

#Source finding
include_directories("include")

add_subdirectory(src)
add_subdirectory(apps)


#Documentation
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(UseDoxygen OPTIONAL)
