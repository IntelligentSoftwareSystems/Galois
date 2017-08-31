set(prefix delaunayRefine)

mesh(2DinCubeDelaunay_275000 275000)
mesh(2DinCubeDelaunay_1000000 1000000)
mesh(2DinCubeDelaunay_2500000 2500000)

## 

# NB(ddn): we have common .cpps in common/ too but since we change .h
# definitions, they cannot be shared as a library, so instead just build
# and link directly
include_directories(common)

add_subdirectory(incrementalRefine)
add_subdirectory(g)
