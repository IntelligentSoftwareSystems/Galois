set(prefix nBody)

onSphere(3DonSphere_1000000 3 1000000)
inCube(3DinCube_1000000 3 1000000)
kuzmin(3Dplummer_1000000 3 1000000)

## 

include_directories(common)

add_subdirectory(parallelCK)
