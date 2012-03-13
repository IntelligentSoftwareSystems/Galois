set(prefix delaunayRefine)
set(has_common "YES")

mesh(2DinCubeDelaunay_275000 275000)
mesh(2DinCubeDelaunay_1000000 1000000)
mesh(2DinCubeDelaunay_2500000 2500000)

## 

include_directories(common)
add_library(pbbs-${prefix}-common common/topologyFromTri.C)

add_subdirectory(incrementalRefine)
