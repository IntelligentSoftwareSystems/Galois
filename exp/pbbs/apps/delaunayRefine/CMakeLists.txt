set(prefix delaunayRefine)
set(has_common "NO")

mesh(2DinCubeDelaunay_275000 275000)
mesh(2DinCubeDelaunay_1000000 1000000)
mesh(2DinCubeDelaunay_2500000 2500000)

## 

add_subdirectory(incrementalRefine)
add_subdirectory(ndIncrementalRefine)
add_subdirectory(nd2IncrementalRefine)
