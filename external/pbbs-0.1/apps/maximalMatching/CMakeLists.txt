set(prefix maximalMatching)

rmat(rMatGraph_E_5_10000000 "YES" 10000000 50000000)
rand(randLocalGraph_E_5_10000000 "YES" 10000000 50000000)
grid(3Dgrid_E_10000000 "YES" 3 10000000)

## 

include_directories(common)

add_subdirectory(incrementalMatching)
add_subdirectory(ndMatching)
add_subdirectory(serialMatching)
