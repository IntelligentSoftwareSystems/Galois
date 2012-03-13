set(prefix delaunayTriangulation)
set(has_common "YES")

inCube(2DinCube_10M 2 10000000)
kuzmin(2Dkuzmin_10M 2 10000000)

## 

# Broken for -O3
if(CMAKE_BUILD_TYPE MATCHES "Release")
  add_definitions(-O2)
endif()

include_directories(common)
add_library(pbbs-${prefix}-common common/topologyFromTri.C)

add_subdirectory(incrementalDelaunay)
add_subdirectory(serialDelaunay)
