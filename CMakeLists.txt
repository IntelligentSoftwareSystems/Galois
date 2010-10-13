cmake_minimum_required (VERSION 2.6)
project (Galois)

SET(Boost_ADDITIONAL_VERSIONS "1.44" "1.44.0")
find_package( Boost 1.44.0 )
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG          "-g")

set(CMAKE_BUILD_TYPE Release)

include_directories("include")

add_subdirectory(src)
add_subdirectory(apps)
