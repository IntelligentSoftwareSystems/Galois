cmake_minimum_required (VERSION 2.6)
project (Galois)

SET(Boost_ADDITIONAL_VERSIONS "1.44" "1.44.0")
find_package( Boost 1.44.0 REQUIRED )
include_directories(${Boost_INCLUDE_DIRS})

find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG")
#set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -g")
#set(CMAKE_CXX_FLAGS_RELEASE        "-DNDEBUG -pg")

set(CMAKE_CXX_FLAGS_DEBUG          "-g")

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

#set(CMAKE_CXX_COMPILER clang++)
#set(CMAKE_CXX_COMPILER /opt/intel/Compiler/11.1/073/bin/intel64/icpc)

include_directories("include")

add_subdirectory(src)
add_subdirectory(apps)


#Documentation
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(UseDoxygen OPTIONAL)
