set(prefix nearestNeighbors)

inCube(2DinCube_10M 2 10000000)
kuzmin(2Dkuzmin_10M 2 10000000)
inCube(3DinCube_10M 3 10000000)
onSphere(3DonSphere_10M 3 10000000)
kuzmin(3Dplummer_10M 3 10000000)

## 

# Broken for -O3
if(CMAKE_BUILD_TYPE MATCHES "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

include_directories(common)

add_subdirectory(octTree2Neighbors)
add_subdirectory(octTreeNeighbors)
