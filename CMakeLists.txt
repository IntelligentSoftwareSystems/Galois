cmake_minimum_required (VERSION 2.6)
project (Galois)

#Options
SET( PTHREAD ON)
#SET( CRAY ON )
SET( WITH_VTUNE ON )
#SET( ASSERT_LOCKS ON )
SET( WITH_DEBUG_INFO ON )
#SET( WITH_WALL ON)
SET( WITH_NO_ASSERTS ON )

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

if (ASSERT_LOCKS)
  add_definitions(-DASSERT_LOCKS)
endif(ASSERT_LOCKS)

if (WITH_DEBUG_INFO)
  add_definitions("-g")
endif(WITH_DEBUG_INFO)

if (WITH_WALL)
  add_definitions("-Wall -Wno-sign-compare")
endif(WITH_WALL)

if(WITH_NO_ASSERTS)
  add_definitions("-DNDEBUG")
endif(WITH_NO_ASSERTS)

#Flags

set(CMAKE_CXX_FLAGS_RELEASE         "-O3")

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

#set(CMAKE_CXX_COMPILER clang++)
#set(CMAKE_CXX_COMPILER /opt/intel/Compiler/11.1/072/bin/intel64/icpc)
#These help the intel compiler
#add_definitions("-wd981 -wd383 -wd869 -wd2196")

#Source finding
include_directories("include")

add_subdirectory(src)
add_subdirectory(apps)


#Documentation
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(UseDoxygen OPTIONAL)
