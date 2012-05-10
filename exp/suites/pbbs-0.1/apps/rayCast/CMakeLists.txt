set(prefix rayCast)

genRays(angelTriangles angelRays)
genRays(dragonTriangles dragonRays)
genRays(happyTriangles happyRays)

## 

include_directories(common)

add_subdirectory(kdTree)
