cmake_minimum_required (VERSION 2.6)
project (delaunayrefinement)

SET(Boost_ADDITIONAL_VERSIONS "1.44" "1.44.0")
find_package( Boost 1.44.0 )
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()


include_directories("../../include")
add_executable(delaunayrefinement DelaunayRefinement.cpp)

