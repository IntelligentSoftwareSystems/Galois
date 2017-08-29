set(prefix maximalIndependentSet)

rmat(rMatGraph_J_5_10000000 "NO" 10000000 50000000)
rand(randLocalGraph_J_5_10000000 "NO" 10000000 50000000)
grid(3Dgrid_J_10000000 "NO" 3 10000000)

## 

include_directories(common)
add_subdirectory(g)
add_subdirectory(incrementalMIS)
add_subdirectory(pincrementalMIS)
add_subdirectory(ndMIS)
add_subdirectory(serialMIS)
