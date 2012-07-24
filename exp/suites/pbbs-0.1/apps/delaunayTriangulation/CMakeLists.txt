set(prefix delaunayTriangulation)

inCube(2DinCube_10M 2 10000000)
kuzmin(2Dkuzmin_10M 2 10000000)

## 

# Broken for -O3
#if(CMAKE_BUILD_TYPE MATCHES "Release")
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
#endif()

# NB(ddn): we have common .cpps in common/ too but since we change .h
# definitions, they cannot be shared as a library, so instead just build
# and link directly
include_directories(common)

add_subdirectory(g)
add_subdirectory(incrementalDelaunay)
add_subdirectory(serialDelaunay)
