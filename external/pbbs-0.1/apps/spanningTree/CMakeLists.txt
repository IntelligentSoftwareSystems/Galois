set(prefix spanningTree)

rmat(rMatGraph_E_5_10000000 "YES" 10000000 50000000)
rand(randLocalGraph_E_5_10000000 "YES" 10000000 50000000)
grid(2Dgrid_E_10000000 "YES" 2 10000000)

## 

include_directories(common)

add_subdirectory(incrementalST)
add_subdirectory(ndST)
#add_subdirectory(newST)
add_subdirectory(serialST)
